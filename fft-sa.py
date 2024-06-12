import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

class FFTDifference:
    def __init__(self, fft_file1, fft_file2, output_file):
        self.fft_file1 = fft_file1
        self.fft_file2 = fft_file2
        self.output_file = output_file
        self.fft_data1 = self.load_fft_data(fft_file1)
        self.fft_data2 = self.load_fft_data(fft_file2)
        if self.fft_data1.size > 0 and self.fft_data2.size > 0:
            self.calculate_difference()
            self.save_difference_data()
        else:
            print("FFTデータが正しく読み込まれませんでした。")

    def load_fft_data(self, file_path):
        try:
            data = pd.read_csv(file_path).values
            print(f"Loaded FFT data from {file_path}, shape: {data.shape}")
        except Exception as e:
            print(f"FFTデータの読み込み中にエラーが発生しました: {e}")
            data = np.array([])
        return data

    def calculate_difference(self):
        freq1 = self.fft_data1[:, 0]
        amp1 = self.fft_data1[:, 1]
        freq2 = self.fft_data2[:, 0]
        amp2 = self.fft_data2[:, 1]

        # 近い周波数同士を比較するためのツリー構造を作成
        tree1 = cKDTree(freq1[:, np.newaxis])
        tree2 = cKDTree(freq2[:, np.newaxis])

        common_freq = []
        amp_diff = []

        # 周波数1の各点に最も近い周波数2の点を見つけて差分を計算
        for f1, a1 in zip(freq1, amp1):
            dist, idx = tree2.query([[f1]])
            if dist[0] < 0.01:  # 許容誤差の設定（必要に応じて調整）
                f2 = freq2[idx[0]]
                a2 = amp2[idx[0]]
                common_freq.append(f1)
                amp_diff.append(a1 - a2)

        self.difference_data = np.vstack((common_freq, amp_diff)).T
        print(f"Calculated difference data shape: {self.difference_data.shape}")

    def save_difference_data(self):
        try:
            np.savetxt(self.output_file, self.difference_data, delimiter=",", fmt='%10.5f', header="Frequency,AmplitudeDifference", comments='')
            print(f"Difference data saved to {self.output_file}")
        except Exception as e:
            print(f"データの保存中にエラーが発生しました: {e}")

def main():
    fft_file1 = 'csv\\' + 'down-fft_output.csv'  # 最初のFFT結果のファイルパスを指定
    fft_file2 =  'csv\\' + 'up-fft_output.csv'  # 2つ目のFFT結果のファイルパスを指定
    output_file =  'csv\\' + 'fft_difference_output.csv'  # 差分を保存するファイルのベース名を指定

    fft_diff = FFTDifference(fft_file1, fft_file2, output_file)

if __name__ == '__main__':
    main()
