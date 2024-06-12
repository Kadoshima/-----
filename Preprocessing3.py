import csv
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense

class InData:
    label = 1  # クラスプロパティとしてのラベル（実数）
    InData = np.array([])  # データプロパティ

    deletelen = 2  # seconds to cut from start and end
    samplingRate = 60  # Hz
    segment_length = 5  # seconds
    step_length = 2  # step size in seconds

    def __init__(self, url, label, output_name) -> None:
        self.label = int(label)  # ラベルを整数に変換
        self.output_name = output_name
        with open(url, encoding='utf8', newline='') as f:
            csvreader = csv.reader(f, delimiter=',')
            data = [row for row in csvreader]
            self.InData = np.array(data, dtype=float)
        print(f"Loaded data length: {len(self.InData)}")
        self.InData[:, 0] = self.InData[:, 0] / 1e3  # タイムスタンプをミリ秒から秒に変換
        self.InData[:, 0] = self.InData[:, 0] - self.InData[0, 0]  # タイムスタンプを0秒から始まるように調整
        print(f"First few rows of adjusted data: {self.InData[:5]}")
        self.resample_data(self.samplingRate)
        self.trim_data(self.deletelen)
        print(f"Data length after trimming: {len(self.InData)}")

    def resample_data(self, new_rate):
        timestamps = self.InData[:, 0]
        start_time = timestamps[0]
        end_time = timestamps[-1]
        num_samples = int((end_time - start_time) * new_rate)
        new_timestamps = np.linspace(start_time, end_time, num_samples)

        interp_x = interp1d(timestamps, self.InData[:, 1], kind='linear')
        interp_y = interp1d(timestamps, self.InData[:, 2], kind='linear')
        interp_z = interp1d(timestamps, self.InData[:, 3], kind='linear')

        self.InData = np.vstack((
            new_timestamps,
            interp_x(new_timestamps),
            interp_y(new_timestamps),
            interp_z(new_timestamps)
        )).T

    def trim_data(self, trim_seconds):
        trim_samples = trim_seconds * self.samplingRate
        if len(self.InData) > 2 * trim_samples:
            self.InData = self.InData[trim_samples:-trim_samples]
        else:
            print("トリムするデータが不足しています。データ長が十分ではないため、トリミングをスキップします。")
            self.InData = self.InData  # トリミングをスキップ

    def segment_data(self):
        segment_samples = self.segment_length * self.samplingRate
        step_samples = self.step_length * self.samplingRate
        segments = []

        start = 0
        while start + segment_samples <= len(self.InData):
            segment = self.InData[start:start + segment_samples]
            segments.append(segment)
            start += step_samples

        return segments

    def save_segments(self, segments, output_dir):
        os.makedirs(output_dir, exist_ok=True)  # 出力ディレクトリが存在しない場合は作成
        for i, segment in enumerate(segments):
            filename = f"{output_dir}/{self.output_name}_{i}.csv"
            np.savetxt(filename, segment[:, 1:], delimiter=",", header='', comments='', fmt='%10.5f')

    def print_segment(self, segment):
        plt.figure(figsize=(10, 6))
        plt.plot(segment[:, 0], segment[:, 1], label='X axis')
        plt.plot(segment[:, 0], segment[:, 2], label='Y axis')
        plt.plot(segment[:, 0], segment[:, 3], label='Z axis')
        plt.xlabel('Time [s]')
        plt.ylabel('Acceleration')
        plt.title('Segment of Acceleration Sensor Data')
        plt.legend()
        plt.grid()
        plt.show()

def __main__():
    file_path = 'きつい坂上り\きつい上り坂02-K.csv'
    output_name = 'きつい上り坂02'
    data = InData(file_path, 2, output_name)
    if data is None:
        print('データの読み込みに失敗しました')
        return

    segments = data.segment_data()

    # セグメントを保存
    output_dir = 'C:\\Users\\tp240\\OneDrive\\デスクトップ\\実験データ\\きつい坂上り\\加工済みデータ'
    data.save_segments(segments, output_dir)

if __name__ == '__main__':
    __main__()
