import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os
import scipy.io


class Utils:
    
    @staticmethod
    def merge_and_label_csv_files(td_csv, adhd_csv, asd_csv):
        """
        ADHD, ASD, TD csv 파일을 읽어와서 각 데이터에 label을 추가하고,
        하나의 DataFrame으로 병합하는 함수입니다.

        Args:
        - td_csv (str): TD 데이터 경로
        - adhd_csv (str): ADHD 데이터 경로
        - asd_csv (str): ASD 데이터 경로


        Returns:
        - pandas.DataFrame: 세 데이터를 병합한 최종 DataFrame
        """

        # CSV 파일들을 DataFrame으로 불러오기
        df_td = pd.read_csv(td_csv)
        df_adhd = pd.read_csv(adhd_csv)
        df_asd = pd.read_csv(asd_csv)


        # 각 DataFrame에 label 컬럼 추가
        df_td['label'] = 0    # TD는 label 0
        df_adhd['label'] = 1   # ADHD는 label 1
        df_asd['label'] = 2    # ASD는 label 2

        # DataFrame 병합
        final_df = pd.concat([df_td, df_adhd, df_asd], ignore_index=True)

        return final_df

    @staticmethod
    def filter_rows_based_on_config(df_input, config):
        """
        config DataFrame에 있는 '연구번호'를 사용하여 'Zpos_' 접두어와 '_' 접미어가 붙은 문자열로
        시작하는 행들만 남기고 새로운 DataFrame을 반환하는 함수입니다. 또한 패턴이 존재하면 출력합니다.

        Args:
        - df_input (pandas.DataFrame): 입력 데이터프레임
        - config (pandas.DataFrame): 연구번호를 포함한 데이터프레임

        Returns:
        - pandas.DataFrame: 조건에 맞는 행들만 남긴 새로운 DataFrame
        """
        filtered_df = pd.DataFrame()  # 빈 DataFrame 생성

        # 각 연구번호에 대해 시작 패턴을 만들고 필터링
        loop = tqdm(config['연구번호'])
        for num in loop:
            pattern = 'ROIval_AAL3NIHPD_' + str(num) + '_'

            # 패턴이 존재하는지 확인 후 출력
            if df_input['Unnamed: 0'].str.startswith(pattern, na=False).any():
                loop.set_description(f"Pattern found: {pattern}")
            else:
                loop.set_description(f"Pattern not found: {pattern}")
            # 해당 패턴으로 시작하는 행들을 필터링해서 추가
            filtered_df = pd.concat([filtered_df, df_input[df_input['Unnamed: 0'].str.startswith(pattern, na=False)]])

        return filtered_df

    @staticmethod
    def edge(data, filename):
        A = np.zeros((166, 166))

        # Set the given coordinates to 1
        for (i, j) in data:
            A[i, j] = 1  # Adjusting for 0-based indexing

        # Create the symmetric matrix A + A.T
        A_sym = A + A.T

        # Plot the symmetric matrix
        plt.figure(figsize=(8, 8))
        plt.imshow(A_sym, cmap='jet', interpolation='none')
        plt.title('Edge Matrix')
        plt.colorbar()
        plt.show()

        # Save the matrix to an Excel file
        csv_data = pd.DataFrame(A_sym)
        csv_data.to_csv(filename)


        # Remove the first row and multiply the remaining data by 5
        modified_data = csv_data.iloc[1:, :] * 5

        # Create the edge file name
        name, _ = os.path.splitext(os.path.basename(filename))
        edge_filename = os.path.join(os.path.dirname(filename), f"{name}.edge")

        # Save the modified data to an edge file
        with open(edge_filename, "w") as f:
            for row in modified_data.itertuples(index=False, name=None):
                f.write("\t".join(map(str, row)) + "\n")

        print(f"Edge file created with values multiplied by 5 (excluding first row): {edge_filename}")
        
    @staticmethod
    def get_coordinates(index):
        n = 166  # AAL3
        row = 0
        while index > row:
            index -= row
            row += 1

        col = index - 1
        return row, col
    

    @staticmethod
    def __extract_lt(file_path):
        """
        .mat 파일에서 여러 개의 166x166 크기의 correlation matrix를 읽고, 
        각 matrix의 하삼각행렬에 위치한 값을 추출한 후 pandas Series 리스트로 반환합니다.

        Args:
        - file_path (str): .mat 파일 경로

        Returns:
        - list of pandas.Series: 각 matrix의 하삼각행렬 값들로 구성된 Series들의 리스트
        """

        # .mat 파일에서 데이터를 읽음
        mat = scipy.io.loadmat(file_path)

        # .mat 파일에서 matrix가 있는 변수명을 모두 탐색
        matrix_keys = [key for key in mat.keys() if not key.startswith('__')]

        all_series = []

        # 각 key에 해당하는 matrix 처리
        for key in matrix_keys:
            matrix = mat[key]

            # 166x166 행렬인지 확인
            if matrix.shape == (166, 166):
                lower_triangular_values = []

                # 하삼각행렬에서 값들을 열별로 추출
                for col in range(166):
                    for row in range(col + 1, 166):
                        lower_triangular_values.append(matrix[row, col])

                # Pandas Series로 변환 (파일 이름과 key를 시리즈의 이름으로 사용)
                series_name = f"{os.path.basename(file_path)}_{key}"
                series = pd.Series(lower_triangular_values, name=series_name)
                all_series.append(series)

        return all_series

    @staticmethod
    def process_folder_to_csv(root_folder, output_csv):
        """
        주어진 폴더를 재귀적으로 순회하면서 .mat 파일을 찾아 하삼각행렬 데이터를 추출하고,
        모든 시리즈를 합친 후 DataFrame을 생성하여 CSV 파일로 저장하는 함수입니다.

        Args:
        - root_folder (str): 순회할 폴더 경로
        - output_csv (str): 최종 저장할 CSV 파일 경로
        """
        all_series = []

        # 폴더 순회
        for dirpath, dirnames, filenames in os.walk(root_folder):
            for file in filenames:
                if file.endswith('.mat'):
                    file_path = os.path.join(dirpath, file)
                    try:
                        # 시리즈 리스트 추출
                        series_list = Utils.__extract_lt(file_path)
                        # 리스트 확장 (extend 사용)
                        all_series.extend(series_list)
                    except Exception as e:
                        print(f"Error processing file {file_path}: {e}")

        # 모든 시리즈를 하나의 DataFrame으로 결합
        if all_series:
            df = pd.concat(all_series, axis=1).T
            df.to_csv(output_csv, index=True)
            print(f"CSV 파일이 성공적으로 저장되었습니다: {output_csv}")
        else:
            print("처리할 .mat 파일이 없습니다.")
            
    @staticmethod
    def plotpkl(data_list):
        plt.figure(figsize=(10, 7))

        colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'cyan', 'magenta'] #원하면 추가

        for idx, dct in enumerate(data_list):
            label = next(iter(dct.keys()))
            data1 = next(iter(dct.values()))
            cmp = data1['compare']
            color = colors[idx % len(colors)]


            plt.errorbar(
                range(1, data1['max_features'] + 1), 
                data1['validation_mean'],
                yerr=data1['validation_std'], 
                label=f'{label}, ({cmp.upper()})', 
                fmt='-o', 
                capsize=5, 
                elinewidth=1,
                alpha=0.6,
                ecolor=color, 
                markerfacecolor=color, 
                markeredgewidth=1,
                markersize=4,
                linewidth=1,
                color=color 
            )


            plt.plot(range(1, data1['max_features'] + 1), 
                     data1['perm_accuracies'], 
                     label=f'Permutation Test ({cmp.upper()})', 
                     linestyle='--', 
                     linewidth=1, 
                     color=color)


        plt.ylim([0.4, 1])
        plt.title('fc-MVPA Results Comparison')
        plt.xlabel('Number of Features')
        plt.ylabel('Validation Accuracy')
        plt.legend(loc='lower center', fontsize='small', bbox_to_anchor=(0.5, -0.3), ncol=3)
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
    
    @staticmethod
    def split_dataframe_two_classes(df, label_col, class1, class2, test_size=0.2):
        """
        두 개의 클래스(class1, class2)를 사용하여 데이터를 나누고, 
        같은 파일 그룹은 같은 train/test 세트에 속하게 하는 함수입니다.

        Args:
        - df (pandas.DataFrame): 입력 데이터프레임
        - label_col (str): 레이블 컬럼명
        - class1 (int): 첫 번째 클래스 (예: 0)
        - class2 (int): 두 번째 클래스 (예: 1 또는 2)
        - test_size (float): 테스트 데이터 비율 (기본값: 0.2)

        Returns:
        - pandas.DataFrame: train 데이터셋
        - pandas.DataFrame: test 데이터셋
        """

        # Unnamed: 0에서 '.mat' 이전까지 추출하여 그룹화할 새로운 열 생성
        df['file_group'] = df['Unnamed: 0'].str.extract(r'(.*\.mat)')

        # 지정된 클래스들에 해당하는 데이터만 추출
        df_class1 = df[df[label_col] == class1]
        df_class2 = df[df[label_col] == class2]

        # 각 클래스를 각각 그룹화하여 split
        df_class1_groups = df_class1.groupby('file_group')
        df_class2_groups = df_class2.groupby('file_group')

        # 각 그룹에서 하나의 대표 행을 기준으로 train/test split
        class1_group_keys = df_class1['file_group'].unique()
        class2_group_keys = df_class2['file_group'].unique()

        class1_train_keys, class1_test_keys = train_test_split(class1_group_keys, test_size=test_size, random_state=42)
        class2_train_keys, class2_test_keys = train_test_split(class2_group_keys, test_size=test_size, random_state=42)

        # train/test 데이터셋 구성
        train_data = pd.concat([df_class1[df_class1['file_group'].isin(class1_train_keys)],
                                df_class2[df_class2['file_group'].isin(class2_train_keys)]])

        test_data = pd.concat([df_class1[df_class1['file_group'].isin(class1_test_keys)],
                               df_class2[df_class2['file_group'].isin(class2_test_keys)]])

        # 'file_group' 열 제거
        train_data = train_data.drop(columns=['file_group'])
        test_data = test_data.drop(columns=['file_group'])

        return train_data, test_data

