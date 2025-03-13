import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from obspy.clients.fdsn import Client
from obspy import UTCDateTime, Stream, read
from obspy.geodetics import locations2degrees, gps2dist_azimuth
from obspy.taup import TauPyModel

###############################################################################
# 1. FDSN 클라이언트 및 TauP 모델 설정
###############################################################################
client_iris = Client("IRIS")
model = TauPyModel(model="iasp91")

###############################################################################
# 2. 지진 검색 범위 설정 및 이벤트 조회
###############################################################################
starttime_input = "2013-05-24T04:44:48.980000Z"
endtime_input   = "2013-05-24T12:44:48.980000Z"
min_magnitude_input = '8'

starttime = UTCDateTime(starttime_input) - 60
endtime   = UTCDateTime(endtime_input)   + 120
min_magnitude = float(min_magnitude_input)

cat = client_iris.get_events(
    starttime=starttime,
    endtime=endtime,
    minmagnitude=min_magnitude,
    catalog="NEIC PDE"
)

events = []
print(f'\n{len(cat)} Event(s) in Catalog : ')
for i, event in enumerate(cat):
    origin = event.preferred_origin() or event.origins[0]
    magnitude = event.preferred_magnitude() or event.magnitudes[0]
    print(f"{i + 1}. {origin.time} | {origin.latitude:+.3f}, {origin.longitude:+.3f} "
          f"| {magnitude.mag:.1f} {magnitude.magnitude_type} | {origin.evaluation_mode}")
    events.append((origin.time, origin.latitude, origin.longitude,
                   magnitude.mag, magnitude.magnitude_type, origin.evaluation_mode))

# 이벤트 선택
choice = 1 - 1
if 0 <= choice < len(events):
    selected_event = events[choice]
    latitude  = selected_event[1]
    longitude = selected_event[2]
    print(f"\n선택된 이벤트: {selected_event[0]} | {selected_event[1]:+.3f}, "
          f"{selected_event[2]:+.3f} | {selected_event[3]:.1f} {selected_event[4]} "
          f"| {selected_event[5]}")
    print(f"진앙의 경도: {longitude}, 진앙의 위도: {latitude}")
else:
    print("유효하지 않은 선택입니다.")
    exit()

###############################################################################
# 3. 관측소 파일 읽어오기 (GSN_Stations.csv)
###############################################################################
stations_file = r"D:\PyCharm\Geophysics\src\kquake_stations.csv"
stations_df = pd.read_csv(stations_file)

###############################################################################
# 4. 진원-관측소 각거리 계산 & CSV 저장 (Distance_in_Degrees)
###############################################################################
if "Distance_in_Degrees" not in stations_df.columns:
    stations_df["Distance_in_Degrees"] = np.nan

distances = []
for i, row in stations_df.iterrows():
    if pd.isna(row["Distance_in_Degrees"]):
        station_lat = row["Latitude"]
        station_lon = row["Longitude"]
        dist_deg = locations2degrees(latitude, longitude, station_lat, station_lon)
        distances.append(dist_deg)
    else:
        distances.append(row["Distance_in_Degrees"])

stations_df["Distance_in_Degrees"] = distances

###############################################################################
# 5. 각거리로 관측소 선별
###############################################################################
min_angle_distance = 10
max_angle_distance = 80
filtered_stations = stations_df[
    (stations_df["Distance_in_Degrees"] >= min_angle_distance) &
    (stations_df["Distance_in_Degrees"] <= max_angle_distance)
]
print(f"\n각거리 {min_angle_distance} ~ {max_angle_distance} 사이 관측소 : ")
print(filtered_stations)

###############################################################################
# 6. P, PcP, S, ScS, ScP, PcS 위상 도착 시간 계산
###############################################################################
event_time   = selected_event[0]
source_depth = 598.1

arrivals_P    = []
arrivals_PcP  = []
arrivals_S    = []
arrivals_ScS  = []
arrivals_ScP  = []
arrivals_PcS  = []

for _, row in stations_df.iterrows():
    dist_deg = row["Distance_in_Degrees"]
    try:
        travel_times = model.get_travel_times(
            source_depth_in_km=source_depth,
            distance_in_degree=dist_deg,
            phase_list=["P", "PcP", "S", "ScS", "ScP", "PcS"]
        )

        p_time    = next((tt.time for tt in travel_times if tt.name == "P"),    None)
        pcp_time  = next((tt.time for tt in travel_times if tt.name == "PcP"),  None)
        s_time    = next((tt.time for tt in travel_times if tt.name == "S"),    None)
        scs_time  = next((tt.time for tt in travel_times if tt.name == "ScS"),  None)
        scp_time  = next((tt.time for tt in travel_times if tt.name == "ScP"),  None)
        pcs_time  = next((tt.time for tt in travel_times if tt.name == "PcS"),  None)

        arrivals_P.append(round(p_time, 6)   if p_time   else np.nan)
        arrivals_PcP.append(round(pcp_time, 6) if pcp_time else np.nan)
        arrivals_S.append(round(s_time, 6)   if s_time   else np.nan)
        arrivals_ScS.append(round(scs_time,6) if scs_time else np.nan)
        arrivals_ScP.append(round(scp_time,6) if scp_time else np.nan)
        arrivals_PcS.append(round(pcs_time,6) if pcs_time else np.nan)

    except:
        arrivals_P.append(np.nan)
        arrivals_PcP.append(np.nan)
        arrivals_S.append(np.nan)
        arrivals_ScS.append(np.nan)
        arrivals_ScP.append(np.nan)
        arrivals_PcS.append(np.nan)

stations_df["arrival_time_P"]    = arrivals_P
stations_df["arrival_time_PcP"]  = arrivals_PcP
stations_df["arrival_time_S"]    = arrivals_S
stations_df["arrival_time_ScS"]  = arrivals_ScS
stations_df["arrival_time_ScP"]  = arrivals_ScP
stations_df["arrival_time_PcS"]  = arrivals_PcS

stations_df.to_csv(stations_file, index=False)

print(f'\n이벤트 발생 시각 : {event_time}')
print(f'파형 다운로드 시작~종료 시각 : {starttime} ~ {endtime}')

###############################################################################
# 7. 파형 다운로드, 전처리 => merged_stream 만들기
###############################################################################
cache_dir = "waveform_cache"
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)

merged_stream = Stream()

for _, station in filtered_stations.iterrows():
    network_name = station["Network"]
    station_name = station["Station"]
    station_lat  = station["Latitude"]
    station_lon  = station["Longitude"]

    try:
        distance_in_degrees = locations2degrees(latitude, longitude, station_lat, station_lon)
        print(f'{network_name}, {station_name}, Distance: {distance_in_degrees:.2f} deg')

        start_str = starttime.strftime("%Y%m%dT%H%M%S")
        end_str   = endtime.strftime("%Y%m%dT%H%M%S")
        cache_filename = os.path.join(
            cache_dir, f"{network_name}_{station_name}_{start_str}_{end_str}.pkl"
        )

        # 캐시 파일이 있으면 로드, 없으면 다운로드
        if os.path.exists(cache_filename):
            print(f"  -> 캐시 파일에서 파형 로드: {cache_filename}")
            st = read(cache_filename, format='PICKLE')
        else:
            print(f"  -> 파형 다운로드 진행: {network_name} {station_name}")
            try:
                st = client_iris.get_waveforms(
                    network=network_name, station=station_name,
                    location="00", channel="BH?",
                    starttime=starttime, endtime=endtime,
                    attach_response=True
                )
            except Exception as e:
                print(f"  -> 데이터 가져오기 실패: {e}")
                continue

            st.write(cache_filename, format='PICKLE')
            print(f"  -> 캐시 파일 저장: {cache_filename}")

        # 전처리 과정
        st.remove_response(output="VEL")
        st.taper(type="cosine", max_percentage=0.05)
        st.detrend(type="demean")
        st.detrend(type="linear")
        st.filter("bandpass", freqmin=0.01, freqmax=0.25)
        st.normalize()

        for tr in st:
            tr.stats.coordinates = {"latitude": station_lat, "longitude": station_lon}

        # ZNE -> LQT 회전 위한 방위각 설정
        for tr_ in st:
            ch_ = tr_.stats.channel.upper()
            if ch_.endswith("Z"):
                tr_.stats.azimuth = 0.0
                tr_.stats.inclination = 0.0
            elif ch_.endswith("N"):
                tr_.stats.azimuth = 0.0
                tr_.stats.inclination = 90.0
            elif ch_.endswith("E"):
                tr_.stats.azimuth = 90.0
                tr_.stats.inclination = 90.0

        _, azimuth, back_azimuth = gps2dist_azimuth(latitude, longitude,
                                                    station_lat, station_lon)
        st.rotate(method="ZNE->LQT", back_azimuth=back_azimuth)

        merged_stream += st

    except Exception as e:
        print(f"데이터 가져오기 실패 : {network_name}, {station_name}. 오류 : {e}")
        continue

for tr in merged_stream:
    station_row = stations_df.loc[stations_df["Station"] == tr.stats.station]
    if not station_row.empty:
        tr.stats.distance = station_row["Distance_in_Degrees"].values[0]

print(f"\n총 {len(merged_stream)}개의 Trace가 포함된 Stream")

###############################################################################
# 8. P=0초로 정렬
###############################################################################

# 8.1) P파를 0초로 설정하고, NaN이면 건너뜀
aligned_stream = Stream()
component_to_plot = "*"

for tr in merged_stream.select(channel=f"*{component_to_plot}"):
    station_name = tr.stats.station
    row_match = stations_df[stations_df["Station"] == station_name]
    if row_match.empty:
        continue

    arrival_p   = row_match["arrival_time_P"].values[0]
    arrival_pcp = row_match["arrival_time_PcP"].values[0]
    arrival_s   = row_match["arrival_time_S"].values[0]
    arrival_scs = row_match["arrival_time_ScS"].values[0]
    arrival_scp = row_match["arrival_time_ScP"].values[0]
    arrival_pcs = row_match["arrival_time_PcS"].values[0]

    # P 도착 시간이 NaN이면 정렬 불가 → 스킵
    if np.isnan(arrival_p):
        continue

    # ----- P=0 정렬 -----
    p_absolute_time = event_time + arrival_p
    shift_in_seconds = -p_absolute_time.timestamp

    tr_aligned = tr.copy()
    tr_aligned.stats.starttime += shift_in_seconds

    def shift_arrival_if_valid(phase_arrival):
        if not np.isnan(phase_arrival):
            return (event_time + phase_arrival).timestamp + shift_in_seconds
        else:
            return None

    tr_aligned.stats.pcparr = shift_arrival_if_valid(arrival_pcp)
    tr_aligned.stats.sarr   = shift_arrival_if_valid(arrival_s)
    tr_aligned.stats.scsarr = shift_arrival_if_valid(arrival_scs)
    tr_aligned.stats.scparr = shift_arrival_if_valid(arrival_scp)
    tr_aligned.stats.pcsarr = shift_arrival_if_valid(arrival_pcs)

    aligned_stream += tr_aligned

print(f"정렬 후, {len(aligned_stream)}개의 Trace를 섹션 플롯 합니다.")

###############################################################################
# 8.2) 섹션(wiggle) 플롯
###############################################################################
fig, ax = plt.subplots(figsize=(10, 12))

tmin, tmax = -100, 10000
ax.axvline(0, color="red", linestyle=":", linewidth=1.5, label="P")

for i, tr in enumerate(sorted(aligned_stream, key=lambda x: x.stats.distance)):
    npts = tr.stats.npts
    dt   = tr.stats.delta
    start_sec = tr.stats.starttime.timestamp
    t_array   = start_sec + np.arange(npts) * dt

    mask = (t_array >= tmin) & (t_array <= tmax)
    if not np.any(mask):
        continue

    t_plot    = t_array[mask]
    data_plot = tr.data[mask].astype(float)

    max_amp = np.max(np.abs(data_plot)) if len(data_plot) else 1
    if max_amp > 0:
        data_plot /= max_amp
    data_plot *= 0.5

    dist_deg = getattr(tr.stats, "distance", np.nan)
    ax.plot(t_plot, dist_deg + data_plot, color="k", linewidth=0.8)

    line_half_height = 1.0

    # PcP(파란색)
    if tr.stats.pcparr is not None:
        xval = tr.stats.pcparr
        y1, y2 = dist_deg - line_half_height, dist_deg + line_half_height
        ax.plot([xval, xval], [y1, y2],
                color="blue", linestyle=":", linewidth=1.2)

    # S(초록색)
    if tr.stats.sarr is not None:
        xval = tr.stats.sarr
        y1, y2 = dist_deg - line_half_height, dist_deg + line_half_height
        ax.plot([xval, xval], [y1, y2],
                color="green", linestyle="--", linewidth=1.2)

    # ScS(보라색)
    if tr.stats.scsarr is not None:
        xval = tr.stats.scsarr
        y1, y2 = dist_deg - line_half_height, dist_deg + line_half_height
        ax.plot([xval, xval], [y1, y2],
                color="magenta", linestyle="--", linewidth=1.2)

    # ScP(노란색)
    if tr.stats.scparr is not None:
        xval = tr.stats.scparr
        y1, y2 = dist_deg - line_half_height, dist_deg + line_half_height
        ax.plot([xval, xval], [y1, y2],
                color="yellow", linestyle="--", linewidth=1.2)

    # PcS(검정색)
    if tr.stats.pcsarr is not None:
        xval = tr.stats.pcsarr
        y1, y2 = dist_deg - line_half_height, dist_deg + line_half_height
        ax.plot([xval, xval], [y1, y2],
                color="black", linestyle="--", linewidth=1.2)

    # 스테이션 라벨(y축 오른쪽)
    ax.text(tmax + 50, dist_deg, f"{tr.stats.station}", va="center", fontsize=8)

ax.set_xlim(tmin, tmax)
dist_vals = [tr.stats.distance for tr in aligned_stream if hasattr(tr.stats, "distance")]
if len(dist_vals) > 0:
    ymin, ymax = min(dist_vals), max(dist_vals)
else:
    ymin, ymax = 0, len(aligned_stream)
ax.set_ylim(ymin - 1, ymax + 1)

ax.set_xlabel("Time (s) relative to P arrival = 0")
ax.set_ylabel("Distance (deg)")
plt.title("Section Plot, aligned at P=0 (PcP=blue, S=green, ScS=magenta, ScP=yellow, PcS=black)")
plt.tight_layout()
plt.show()