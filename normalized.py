import csv
import numpy as np
from sklearn.preprocessing import StandardScaler

class NormalizeData:
    def __init__(self, input_file, output_file):
        self.input_file = input_file
        self.output_file = output_file
        self.data = self.load_data()
        if self.data.size > 0:
            self.calculate_norm()
            self.standardize_amplitude()
            self.save_normalized_data()
        else:
            print("データが読み込まれませんでした。")

    def load_data(self):
        data = []
        try:
            with open(self.input_file, newline='', encoding='utf-8') as csvfile:
                reader = csv.reader(csvfile)
                for row in reader:
                    data.append([float(item) for item in row])
            data = np.array(data)
            print(f"Loaded data shape: {data.shape}")
        except Exception as e:
            print(f"データの読み込み中にエラーが発生しました: {e}")
        return data

    def calculate_norm(self):
        self.norm_data = np.linalg.norm(self.data, axis=1)
        print(f"Calculated norm shape: {self.norm_data.shape}")

    def standardize_amplitude(self):
        scaler = StandardScaler()
        self.norm_data = scaler.fit_transform(self.norm_data.reshape(-1, 1)).flatten()
        print(f"Standardized norm shape: {self.norm_data.shape}")

    def save_normalized_data(self):
        try:
            np.savetxt(self.output_file, self.norm_data, delimiter=",", fmt='%10.5f')
            print(f"Normalized data saved to {self.output_file}")
        except Exception as e:
            print(f"データの保存中にエラーが発生しました: {e}")

def main():
    UpDown = 'up'  # 上り坂の場合は 'up'、下り坂の場合は 'down' を指定
    input_csv = 'csv\\' +  UpDown + '-output_interpolated.csv'  # 読み込むCSVファイルのパスを指定
    output_csv = 'csv\\' +  UpDown + '-normalized_output.csv'  # 保存するファイルのベース名を指定

    normalizer = NormalizeData(input_csv, output_csv)

if __name__ == '__main__':
    main()
