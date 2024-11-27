import numpy as np
import pandas as pd
from scipy.stats import f_oneway, shapiro, mannwhitneyu

class Parametrics:

    @staticmethod
    def f_test(data_csv):
        df = pd.read_csv(data_csv, header=None)
        df.drop(0, inplace=True)

        # Feature와 label 분리
        features = df.iloc[:, 2:-1]  # 마지막 열(label)을 제외한 feature (13695개의 feature)
        labels = df.iloc[:, -1]      # label은 마지막 열 (644개의 label)
        labels = labels.astype('int32')
        print(labels)
        unique_labels = np.unique(labels)

        if len(unique_labels) < 2:
            raise ValueError("There must be at least two unique labels for a two-sample F-test.")

        p_values = []

        # Iterate through each feature (column)
        for feature in features.columns:
            feature_data = features[feature]

            # F-test between label 0 and label 1 (two samples)
            f_value, p_value = f_oneway(feature_data[labels == unique_labels[0]], 
                                        feature_data[labels == unique_labels[1]])

            # Append the p-values
            p_values.append(p_value)

        # Convert to numpy array
        p_values_array = np.array(p_values)
        return p_values_array
    
    @staticmethod
    def normality_test(data_csv):
        # 데이터 로드
        df = pd.read_csv(data_csv, header=None)
        df.drop(0, inplace=True)

        # feature와 label 분리
        features = df.iloc[:, 2:-1]  # 마지막 열(label)을 제외한 feature
        labels = df.iloc[:, -1]      # label은 마지막 열
        labels = labels.astype('int32')

        # 두 개의 고유한 라벨만 사용 (두 샘플)
        unique_labels = np.unique(labels)
        if len(unique_labels) < 2:
            raise ValueError("There must be at least two unique labels for a two-sample normality test.")

        # 결과를 저장할 2 * feature수 크기의 numpy 배열 초기화
        num_features = features.shape[1]
        p_values = np.zeros((2, num_features))

        # 각 feature에 대해 label 0, 1에 대한 정규성 검정 수행
        for feature_idx in range(num_features):
            for label_idx, label_value in enumerate(unique_labels[:2]):  # label 0, 1에 대해
                # 해당 label에 속하는 데이터만 추출하여 feature별로 정규성 검정
                feature_data = features[labels == label_value].iloc[:, feature_idx]

                # Shapiro-Wilk 검정으로 p-value 계산
                _, p_value = shapiro(feature_data)
                p_values[label_idx, feature_idx] = p_value

        return p_values

    
class Nonparametrics:
    
    @staticmethod
    def u_value_array(data_csv, compare):
        """
        지정된 두 그룹 간의 Mann-Whitney U-test 결과인 u-value 리스트를 생성하여 2차원 Numpy 배열을 반환합니다.

        Args:
        - data_csv (str): CSV 파일 경로
        - compare (str): 무엇과 무엇을 비교할지 ('td-adhd', 'td-asd', 'adhd-asd')

        Returns:
        - u_value_array (np.ndarray): LOOCV를 통해 얻은 644 x 13695 크기의 u-value 리스트 배열
        """
        
        df = pd.read_csv(data_csv, header=None)
        df.drop(0, inplace = True)

        features = df.iloc[:, 2:-1]  # 마지막 열(label)을 제외한 feature (13695개의 feature)
        labels = df.iloc[:, -1]     # label은 마지막 열 (644개의 label)
        labels = labels.astype('int32')

        num_subjects = len(labels)


        num_features = features.shape[1]


        if compare == 'td-adhd':
            class_1 = 0  # TD
            class_2 = 1  # ADHD
        elif compare == 'td-asd':
            class_1 = 0  # TD
            class_2 = 2  # ASD
        else:
            raise ValueError("Invalid compare argument. Choose from 'td-adhd', 'td-asd'.")

        # 비교하는 두 그룹(class_1, class_2)에 속하지 않는 데이터 드랍
        filtered_features = features[labels.isin([class_1, class_2])]
        filtered_labels = labels[labels.isin([class_1, class_2])]

        # 비교할 두 클래스의 데이터만 선택
        group_1 = filtered_features[filtered_labels == class_1]
        group_2 = filtered_features[filtered_labels == class_2]
        print(group_1.shape, group_2.shape)

        # 각 피처에 대해 Mann-Whitney U-test 수행
        u_values = []
        for feature_idx in range(num_features):
            feature_1 = group_1.iloc[:, feature_idx]
            feature_2 = group_2.iloc[:, feature_idx]

            # NaN이 포함된 경우 처리 (데이터가 충분하지 않거나 동일한 값일 경우)
            if feature_1.isna().sum() > 0 or feature_2.isna().sum() > 0:
                continue  # 결측값이 있으면 해당 피처를 무시

            # 두 그룹간 Mann-Whitney U-test 수행 (클래스 1 vs 클래스 2)
            try:
                u_value, _ = mannwhitneyu(feature_1, feature_2, alternative='two-sided')
            except Exception as e:
                print(f"Error calculating U-test for feature {feature_idx}: {e}")
                continue

            # u-value가 NaN이면 건너뜀
            if np.isnan(u_value):
                continue

            # 해당 u-value를 u_values 리스트에 저장
            u_values.append(u_value)

        return np.array(u_values)
    
    
    @staticmethod
    def p_value_array(data_csv, compare):
        """
        지정된 두 그룹 간의 Mann-Whitney U-test 결과인 u-value 리스트를 생성하여 2차원 Numpy 배열을 반환합니다.

        Args:
        - data_csv (str): CSV 파일 경로
        - compare (str): 무엇과 무엇을 비교할지 ('td-adhd', 'td-asd', 'adhd-asd')

        Returns:
        - u_value_array (np.ndarray): LOOCV를 통해 얻은 644 x 13695 크기의 u-value 리스트 배열
        """
        # 데이터 불러오기
        df = pd.read_csv(data_csv, header=None)
        df.drop(0, inplace = True)

        # feature와 label 분리
        features = df.iloc[:, 2:-1]  # 마지막 열(label)을 제외한 feature (13695개의 feature)
        labels = df.iloc[:, -1]     # label은 마지막 열 (644개의 label)
        labels = labels.astype('int32')


        num_subjects = len(labels)
        num_features = features.shape[1]



        # 비교할 클래스 설정
        if compare == 'td-adhd':
            class_1 = 0  # TD
            class_2 = 1  # ADHD
        elif compare == 'td-asd':
            class_1 = 0  # TD
            class_2 = 2  # ASD
        elif compare == 'adhd-asd':
            class_1 = 1  # ADHD
            class_2 = 2  # ASD
        else:
            raise ValueError("Invalid compare argument. Choose from 'td-adhd', 'td-asd', 'adhd-asd'.")

        # 비교하는 두 그룹(class_1, class_2)에 속하지 않는 데이터 드랍
        filtered_features = features[labels.isin([class_1, class_2])]
        filtered_labels = labels[labels.isin([class_1, class_2])]

        # 비교할 두 클래스의 데이터만 선택
        group_1 = filtered_features[filtered_labels == class_1]
        group_2 = filtered_features[filtered_labels == class_2]
        print(group_1.shape, group_2.shape)

        # 각 피처에 대해 Mann-Whitney U-test 수행
        p_values = []
        for feature_idx in range(num_features):
            feature_1 = group_1.iloc[:, feature_idx]
            feature_2 = group_2.iloc[:, feature_idx]

            # NaN이 포함된 경우 처리 (데이터가 충분하지 않거나 동일한 값일 경우)
            if feature_1.isna().sum() > 0 or feature_2.isna().sum() > 0:
                continue  # 결측값이 있으면 해당 피처를 무시

            # 두 그룹간 Mann-Whitney U-test 수행 (클래스 1 vs 클래스 2)
            try:
                _, p_value = stats.mannwhitneyu(feature_1, feature_2, alternative='two-sided')
            except Exception as e:
                print(f"Error calculating U-test for feature {feature_idx}: {e}")
                continue

            # u-value가 NaN이면 건너뜀
            if np.isnan(p_value):
                continue

            # 해당 u-value를 u_values 리스트에 저장
            p_values.append(p_value)

        return np.array(p_values)