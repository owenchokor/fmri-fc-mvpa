import numpy as np
import pandas as pd
import random
import pickle
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
from sklearn.model_selection import StratifiedShuffleSplit

# +
class MVPA:
    
    @staticmethod
    def mvpa_fit(train_csv, test_csv, sorted_feature_indices, compare, max_features=100, random_seed=42, params_file='./', n_runs=10):
        """
        MVPA 분석을 수행하고, n-run 만큼 반복하여 cross-validation을 통해 SVM을 학습합니다.
        각 fold마다 validation accuracy를 계산하며, feature importance 순으로 피처를 하나씩 추가하면서
        가장 좋은 성능을 보이는 feature set으로 최종 모델을 학습하여 test set에 대한 성능을 평가합니다.
        마지막에는 permutation test를 통해 label을 랜덤하게 섞고 동일한 절차를 반복합니다.

        Args:
        - train_csv (str): train 데이터 CSV 파일 경로
        - test_csv (str): test 데이터 CSV 파일 경로
        - sorted_feature_indices (np.ndarray): 중요도 순으로 정렬된 피처 인덱스 배열
        - compare (str): 무엇과 무엇을 비교할지 ('td-adhd', 'td-asd')
        - max_features (int): 누적해서 추가할 최대 feature 수
        - random_seed (int): 랜덤 시드 값
        - params_file (str): 실험에서 사용된 파라미터를 저장할 파일 경로 (기본값은 'svm_params.pkl')
        - n_run (int): cross-validation을 반복할 횟수 (기본값은 10)

        Returns:
        - validation_accuracies (dict): 각 fold에 대해 저장된 validation accuracy 리스트
        - test_accuracy (float): test set에 대한 최종 accuracy
        - best_feature_indices (list): 각 fold에 대해 가장 좋은 성능을 보이는 feature set 인덱스 리스트
        """


        random.seed(random_seed)
        np.random.seed(random_seed)


    
        train_df = pd.read_csv(train_csv, header=None)
        train_df.drop(0, inplace=True)
        
        first_roi_column = next(
            (col_idx for col_idx, col in enumerate(train_df.columns) if train_df[col].astype(str).str.startswith("ROIval").any()),
            None
        )


        unnamed_count = first_roi_column + 1

        #print(unnamed_count)
                


        train_features = train_df.iloc[:, unnamed_count:-1]
        train_labels = train_df.iloc[:, -1]
        train_labels = train_labels.astype('int32')
        file_names_train = train_df.iloc[:, unnamed_count-1]
        

        test_df = pd.read_csv(test_csv, header=None)
        test_df.drop(0, inplace = True)


        test_features = test_df.iloc[:, unnamed_count:-1]
        test_labels = test_df.iloc[:, -1]
        test_labels = test_labels.astype('int32')
        file_names_test = test_df.iloc[:, unnamed_count-1]
        

        if compare == 'td-adhd':
            class_1 = 0  # TD
            class_2 = 1  # ADHD
        elif compare == 'td-asd':
            class_1 = 0  # TD
            class_2 = 2  # ASD
        else:
            raise ValueError("Invalid compare argument. Choose from 'td-adhd', 'td-asd'.")


        train_mask = train_labels.isin([class_1, class_2])
        train_features = train_features[train_labels.isin([class_1, class_2])]
        train_labels = train_labels[train_labels.isin([class_1, class_2])]
        file_names_train = file_names_train[train_mask]
        
        test_mask = test_labels.isin([class_1, class_2])
        test_features = test_features[test_labels.isin([class_1, class_2])]
        test_labels = test_labels[test_labels.isin([class_1, class_2])]
        file_names_test = file_names_test[test_mask]
        

        train_labels = np.where(train_labels == class_1, 0, 1)
        test_labels = np.where(test_labels == class_1, 0, 1)
        train_labels = pd.Series(train_labels, index=train_features.index)
        train_features['file_group'] = file_names_train.str.extract(r'(.*\.mat)')
        test_features['file_group'] = file_names_test.str.extract(r'(.*\.mat)')
        group_labels = train_features.groupby('file_group').apply(lambda x: train_labels.loc[x.index].mode()[0])
        
        # 10fold - cv
        sss = StratifiedShuffleSplit(n_splits=n_runs, test_size=0.1, random_state=random_seed)

        validation_accuracies_per_feature = []
        best_feature_indices_per_run = [] 
        test_accuracies = [] 
        confusion_matrices = []
        
        unique_groups = train_features[['file_group']].drop_duplicates()
        
        # Cross-validation 시작
        for run_idx, (train_group_index, val_group_index) in enumerate(sss.split(unique_groups['file_group'], group_labels)):
            train_groups = unique_groups.iloc[train_group_index]['file_group']
            val_groups = unique_groups.iloc[val_group_index]['file_group']

            X_train = train_features[train_features['file_group'].isin(train_groups)].drop(columns=['file_group'])
            X_val = train_features[train_features['file_group'].isin(val_groups)].drop(columns=['file_group'])

            y_train = train_labels[X_train.index]
            y_val = train_labels[X_val.index]


            svm = SVC(kernel='linear')
            accuracies = []
            best_feature_indices = []
            best_run_accuracy = 0

            #  feature를 하나씩 추가하면서 학습
            for i in tqdm(range(1, max_features + 1), desc=f'Run {run_idx + 1}: Training SVM on validation set'):
                selected_features = sorted_feature_indices[:i]
                X_train_selected = X_train.iloc[:, selected_features]
                X_val_selected = X_val.iloc[:, selected_features]


                svm.fit(X_train_selected, y_train)
                y_val_pred = svm.predict(X_val_selected)
                val_acc = accuracy_score(y_val, y_val_pred)
                accuracies.append(val_acc)


                if val_acc > best_run_accuracy:
                    best_run_accuracy = val_acc
                    best_feature_indices = selected_features

            validation_accuracies_per_feature.append(accuracies)
            best_feature_indices_per_run.append(best_feature_indices)
            best_svm = SVC(kernel='linear')
            
            X_train_best = train_features.iloc[:, best_feature_indices]
            best_svm.fit(X_train_best, train_labels)
            X_test_best = test_features.iloc[:, best_feature_indices]
            y_test_pred = best_svm.predict(X_test_best)
            test_accuracy = accuracy_score(test_labels, y_test_pred)
            test_accuracies.append(test_accuracy)
            conf_matrix = confusion_matrix(test_labels, y_test_pred)
            confusion_matrices.append(conf_matrix)
            print(f"Test accuracy with best feature set of run {run_idx + 1}: {test_accuracy:.4f}")
        validation_mean = np.mean(validation_accuracies_per_feature, axis=0)
        validation_std = np.std(validation_accuracies_per_feature, axis=0)

        plt.figure(figsize=(10, 7))
        plt.errorbar(
            range(1, max_features + 1), 
            validation_mean, 
            yerr=validation_std, 
            label='Mean Validation Accuracy', 
            fmt='-o', 
            capsize=5, 
            elinewidth=1, 
            alpha=0.6, 
            ecolor='gray',
            markerfacecolor='blue', 
            markeredgewidth=1,
            markersize=4,
            linewidth=1 
        )

        random_labels = np.random.randint(0, 2, size=len(train_labels))
        svm = SVC(kernel='linear')
        perm_accuracies = []

        for i in tqdm(range(1, max_features + 1), desc='Permutation Test'):
            selected_features = sorted_feature_indices[:i]
            X_train_selected = train_features.iloc[:, selected_features]

            svm.fit(X_train_selected, random_labels)
            y_train_pred = svm.predict(X_train_selected)
            perm_acc = accuracy_score(train_labels, y_train_pred)
            perm_accuracies.append(perm_acc)

        # Permutation test plot
        plt.plot(range(1, max_features + 1), perm_accuracies, label='Permutation Test', linestyle='--', linewidth=1, color='orange')


        plt.ylim([0.4, 1])
        plt.title(f'fc-MVPA results, {compare.upper()}')
        plt.xlabel('Number of Features')
        plt.ylabel('Validation Accuracy')
        plt.legend(loc='lower center', fontsize='small', bbox_to_anchor=(0.5, -0.3), ncol=3)
        plt.grid(True)
        plt.tight_layout()
        plt.show()

 
        mean_test_accuracy = np.mean(test_accuracies)
        std_test_accuracy = np.std(test_accuracies)

        print(f"Mean test accuracy: {mean_test_accuracy:.4f}")
        print(f"Test accuracy standard deviation: {std_test_accuracy:.4f}")

        params = {
            'random_seed': random_seed,
            'compare': compare,
            'max_features': max_features,
            'validation_mean': validation_mean,
            'validation_std': validation_std,
            'test_accuracies': test_accuracies,
            'mean_test_accuracy': mean_test_accuracy,
            'std_test_accuracy': std_test_accuracy,
            'best_feature_indices_per_fold': best_feature_indices_per_run,
            'perm_accuracies': perm_accuracies
        }

        with open(params_file, 'wb') as f:
            pickle.dump(params, f)

        print(f"Parameters saved to {params_file}")

        return validation_mean, validation_std, perm_accuracies, test_accuracies, best_feature_indices_per_run, confusion_matrices



    @staticmethod
    def svm_feature_selection(train_csv, test_csv, sorted_feature_indices, compare, x, random_seed=42):
        """
        주어진 상위 x개의 feature를 사용하여 SVM을 학습하고 test set에 대한 accuracy를 측정하는 함수.

        Args:
        - train_csv (str): train 데이터 CSV 파일 경로
        - test_csv (str): test 데이터 CSV 파일 경로
        - sorted_feature_indices (np.ndarray): 중요도 순으로 정렬된 피처 인덱스 배열
        - compare (str): 비교할 그룹 ('td-adhd', 'td-asd', 'adhd-asd')
        - x (int): 사용할 상위 feature의 개수
        - random_seed (int): 랜덤 시드 값

        Returns:
        - test_accuracy (float): test set에 대한 accuracy
        """


        np.random.seed(random_seed)


        train_df = pd.read_csv(train_csv, header=None)
        train_df.drop(0, inplace=True)


        train_features = train_df.iloc[:, 2:-1]
        train_labels = train_df.iloc[:, -1].astype('int32')


        test_df = pd.read_csv(test_csv, header=None)
        test_df.drop(0, inplace=True)


        test_features = test_df.iloc[:, 2:-1]
        test_labels = test_df.iloc[:, -1].astype('int32')


        if compare == 'td-adhd':
            class_1 = 0  # TD
            class_2 = 1  # ADHD
        elif compare == 'td-asd':
            class_1 = 0  # TD
            class_2 = 2  # ASD
        else:
            raise ValueError("Invalid compare argument. Choose from 'td-adhd', 'td-asd', 'adhd-asd'.")


        train_features = train_features[train_labels.isin([class_1, class_2])]
        train_labels = train_labels[train_labels.isin([class_1, class_2])]

        test_features = test_features[test_labels.isin([class_1, class_2])]
        test_labels = test_labels[test_labels.isin([class_1, class_2])]


        train_labels = np.where(train_labels == class_1, 0, 1)
        test_labels = np.where(test_labels == class_1, 0, 1)


        selected_features = sorted_feature_indices[:x]
        svm = SVC(kernel='linear', random_state=random_seed)


        X_train_selected = train_features.iloc[:, selected_features]
        X_test_selected = test_features.iloc[:, selected_features]


        svm.fit(X_train_selected, train_labels)

        y_test_pred = svm.predict(X_test_selected)
        test_accuracy = accuracy_score(test_labels, y_test_pred)

        print(f"Test accuracy using top {x} features: {test_accuracy:.4f}")

        return test_accuracy
    


