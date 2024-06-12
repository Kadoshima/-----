import csv
import numpy as np
from scipy.interpolate import interp1d

class InData:
    def __init__(self, file_path, output_name):
        self.file_path = file_path
        self.output_name = output_name
        self.InData = self.load_data()
        if self.InData.size > 0:
            self.resample_data()
            self.save_interpolated_data()
        else:
            print("データが読み込まれませんでした。")

    def load_data(self):
        data = []
        try:
            with open(self.file_path, newline='', encoding='utf-8') as csvfile:
                reader = csv.reader(csvfile)
                for row in reader:
                    data.append([float(item) for item in row])
            data = np.array(data)
            print(f"Loaded data shape: {data.shape}")
        except Exception as e:
            print(f"データの読み込み中にエラーが発生しました: {e}")
        return data

    def resample_data(self):
        try:
            timestamps = self.InData[:, 0] / 1e3  # ミリ秒から秒に変換
            x = self.InData[:, 1]
            y = self.InData[:, 2]
            z = self.InData[:, 3]

            start_time = timestamps[0]
            end_time = timestamps[-1]
            num_samples = int((end_time - start_time) * 60)  # Assuming 60 Hz sampling rate

            print(f"Start time: {start_time}, End time: {end_time}, Number of samples: {num_samples}")

            if num_samples <= 0:
                print("Number of samples is not positive. Check the timestamp values.")
                self.interpolated_data = np.array([])  # エラー時には空の配列にする
                return

            new_timestamps = np.linspace(start_time, end_time, num_samples)

            interp_x = interp1d(timestamps, x, kind='linear', fill_value='extrapolate')
            interp_y = interp1d(timestamps, y, kind='linear', fill_value='extrapolate')
            interp_z = interp1d(timestamps, z, kind='linear', fill_value='extrapolate')

            self.interpolated_data = np.vstack((
                interp_x(new_timestamps),
                interp_y(new_timestamps),
                interp_z(new_timestamps)
            )).T
            print(f"Interpolated data shape: {self.interpolated_data.shape}")
            
            # サンプリング周波数の計算
            actual_sampling_rate = calculate_sampling_rate(new_timestamps)
            print(f"Actual sampling rate after interpolation: {actual_sampling_rate:.2f} Hz")
            
        except Exception as e:
            print(f"データの補完中にエラーが発生しました: {e}")
            self.interpolated_data = np.array([])  # エラー時には空の配列にする

    def save_interpolated_data(self):
        try:
            output_file = self.output_name
            if self.interpolated_data.size > 0:
                np.savetxt(output_file, self.interpolated_data, delimiter=",", fmt='%10.5f')
                print(f"Interpolated data saved to {output_file}")
            else:
                print("保存する補完データがありません。")
        except Exception as e:
            print(f"データの保存中にエラーが発生しました: {e}")

def calculate_sampling_rate(timestamps):
    intervals = np.diff(timestamps)
    average_interval = np.mean(intervals)
    sampling_rate = 1 / average_interval
    return sampling_rate

def main():
    UpDown = 'up'  # 上り坂の場合は 'up'、下り坂の場合は 'down' を指定
    input_csv = 'きつい坂上り\きつい上り坂02-K.csv'  # 読み込むCSVファイルのパスを指定
    output_name = 'csv\\' + UpDown + '-output_interpolated.csv'  # 保存するファイルのベース名を指定

    data_processor = InData(input_csv, output_name)

if __name__ == '__main__':
    main()
