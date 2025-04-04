import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from obspy.clients.fdsn import Client
from obspy import UTCDateTime, Stream, read, Inventory
from obspy.geodetics import locations2degrees, gps2dist_azimuth
from obspy.taup import TauPyModel

### 1. Obspy를 이용한 KIGAM Quake FDSN Web Service 이용, 매개변수 정의 ###
client = Client("https://quake.kigam.re.kr/")
client_iris = Client("IRIS")
client.request_headers["X-Open-Api-Token"] = "badGCnPvHtnPNWQL0j8tTTX6XnXsDk"
stations_file = r"D:\PyCharm\Geophysics\src\kquake_stations.csv"

model = TauPyModel(model="iasp91")

### 2. 지진 검색 범위 설정 및 이벤트 조회 ###
## 지진 이벤트 조회 ##
starttime_input = "2024-10-30T08:18:51Z"
endtime_input   = "2024-10-30T19:59:59Z"
min_magnitude_input = '5.9'
max_magnitude_input = '6.0'
source_depth = 509.0

starttime = UTCDateTime(starttime_input) - 120
endtime   = UTCDateTime(endtime_input)   + 120
min_magnitude = float(min_magnitude_input)
max_magnitude = float(max_magnitude_input)

## 지진 이벤트 관련 정보 찾기 ##
cat = client_iris.get_events(
    starttime=starttime,
    endtime=endtime,
    minmagnitude=min_magnitude,
    maxmagnitude=max_magnitude,
    catalog="NEIC PDE"
)

# 지진 이벤트 선택 (1번 이벤트로 자동 선정) #
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

### 3. 데이터 불러오기 ###
stations_df = pd.read_csv(stations_file)

## 각거리, 도착 시간 계산 ##
if "Distance_in_Degrees" not in stations_df.columns:
    stations_df["Distance_in_Degrees"] = np.nan

distances = []
for i, row in stations_df.iterrows():
    if pd.isna(row["Distance_in_Degrees"]):
        station_lat = row["Latitude"]
        station_lon = row["Longitude"]
        # locations2degrees로 각거리 계산
        dist_deg = locations2degrees(latitude, longitude, station_lat, station_lon)
        distances.append(dist_deg)
    else:
        distances.append(row["Distance_in_Degrees"])

stations_df["Distance_in_Degrees"] = distances

print(f"\n 코드에서 사용하는 관측소 : ")
print(stations_df)

## 위상 도착 시간 계산 ##
event_time = selected_event[0] # IRIS에서 가져온 정보 사용
source_depth = source_depth

arrivals_P, arrivals_PcP, arrivals_S, arrivals_ScS, arrivals_PcS, arrivals_ScP = [], [], [], [], [], []

for _, row in stations_df.iterrows():
    dist_deg = row["Distance_in_Degrees"]
    try:
        # 위상 도착 시간 계산
        travel_times = model.get_travel_times(
            source_depth_in_km=source_depth,
            distance_in_degree=dist_deg
        )

        # 각각 해당하는 phase가 있으면 time을 읽고 없으면 np.nan
        p_time    = next((tt.time for tt in travel_times if tt.name == "P"),    None)
        pcp_time  = next((tt.time for tt in travel_times if tt.name == "PcP"),  None)
        s_time    = next((tt.time for tt in travel_times if tt.name == "S"),    None)
        scs_time  = next((tt.time for tt in travel_times if tt.name == "ScS"),  None)
        pcs_time  = next((tt.time for tt in travel_times if tt.name == "PcS"),  None)
        scp_time  = next((tt.time for tt in travel_times if tt.name == "ScP"),  None)

        arrivals_P.append(round(p_time, 6)     if p_time   else np.nan)
        arrivals_PcP.append(round(pcp_time, 6) if pcp_time else np.nan)
        arrivals_S.append(round(s_time, 6)     if s_time   else np.nan)
        arrivals_ScS.append(round(scs_time, 6) if scs_time else np.nan)
        arrivals_PcS.append(round(pcs_time, 6) if pcs_time else np.nan)
        arrivals_ScP.append(round(scp_time, 6) if scp_time else np.nan)


    except Exception as e:
        print(f"[ERROR in get_travel_times] Station={station_name}, dist={dist_deg:.2f}, Exception={e}")

        arrivals_P.append(np.nan)
        arrivals_PcP.append(np.nan)
        arrivals_S.append(np.nan)
        arrivals_ScS.append(np.nan)
        arrivals_PcS.append(np.nan)

# 새로운 컬럼에 저장
stations_df["arrival_time_P"]    = arrivals_P
stations_df["arrival_time_PcP"]  = arrivals_PcP
stations_df["arrival_time_S"]    = arrivals_S
stations_df["arrival_time_ScS"]  = arrivals_ScS
stations_df["arrival_time_PcS"]  = arrivals_PcS
stations_df["arrival_time_ScP"]  = arrivals_ScP


print(f"\n 위상 도착 시간 계산 완료.")

# CSV에 저장
stations_df.to_csv(stations_file, index=False)
print(f"\n 위상 도착 시간 저장 완료.")

## 관측소 메타데이터 다운로드 ##
inventory = client.get_stations(level="response")
inventory.write("KIGAM_KG61_metadata.xml", format="STATIONXML")
print(f"\n 관측소 메타데이터 다운로드 내역 : ")
print(inventory)
print(f"\n 관측소 메타데이터 다운로드 완료.")

## 관측소 파형 다운로드 ##
# 관측소 데이터 cache 저장 #
cache_dir = "waveform_cache"
print(f"\n 캐시 파일에서 데이터 참조 중 ...\n")
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)
    print(f"\n 캐시 파일 생성 중 ...\n")

merged_stream = Stream()

for _, station in stations_df.iterrows():
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
            print(f"  => 캐시 파일에서 파형 로드: {cache_filename}")
            st = read(cache_filename, format='PICKLE')
        else:
            print(f"  => 파형 다운로드 진행: {network_name} {station_name}")
            try:
                st = client.get_waveforms(
                    network=network_name, station=station_name,
                    location="*", channel="BH?",
                    starttime=starttime, endtime=endtime,
                    attach_response=True
                )
            except Exception as e:
                print(f"  => 데이터 가져오기 실패: {e}")
                continue

            st.write(cache_filename, format='PICKLE')
            print(f"  -> 캐시 파일 저장: {cache_filename}")

        # 전처리 과정
        pre_filt = (0.002, 0.005, 10.0, 20.0)
        st.remove_response(inventory=inventory, output="VEL", pre_filt=pre_filt)
        st.taper(type="cosine", max_percentage=0.05)
        st.detrend(type="demean")
        st.detrend(type="linear")
        st.filter("bandpass", freqmin=0.02, freqmax=0.2, corners=4, zerophase=True)
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

        # 관측소 BackAzimuth 계산 => 회전
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

## 물리값 변환 및 관측장비 응답특성 제거 ##
# 메타데이터가 있는 지진파형자료만 추출(extract_metadata_exist_stream 함수 사용) #
def extract_metadata_exist_stream(stream: Stream, inv: Inventory) -> Stream:
    metadata_exist_stream = Stream()
    for trace in stream:
        channel_metadata = inv.select(
            trace.stats.network,
            trace.stats.station,
            trace.stats.location,
            trace.stats.channel,
        )
        if len(channel_metadata) > 0:
            for metadata in channel_metadata[0][0]:
                trimmed_trace = trace.copy()
                trimmed_trace.trim(metadata.start_date, metadata.end_date)
                if trimmed_trace.stats.npts > 0:
                    metadata_exist_stream.append(trimmed_trace)
    return metadata_exist_stream

metadata_exist_stream = extract_metadata_exist_stream(merged_stream, inventory)
sensitivity_removed_stream = metadata_exist_stream.copy().remove_sensitivity(inventory)

# 관측장비 응답특성 제거
response_removed_stream = metadata_exist_stream.copy().remove_response(inventory, output="ACC")

## 2025.03.11 기준 여기까지 완료 ##

###############################################################################
# 7. P=0초로 정렬 후, 정해진 구간에 따라 plot
###############################################################################
# 8.1) P파를 0초로 설정
###############################################################################
aligned_stream = Stream()

component_to_plot = "T"

for tr in merged_stream.select(channel=f"*{component_to_plot}"):
    station_name = tr.stats.station
    row_match = stations_df[stations_df["Station"] == station_name]
    if row_match.empty:
        continue

    arrival_p    = row_match["arrival_time_P"].values[0]
    arrival_pcp  = row_match["arrival_time_PcP"].values[0]
    arrival_s    = row_match["arrival_time_S"].values[0]
    arrival_scs  = row_match["arrival_time_ScS"].values[0]
    arrival_pcs  = row_match["arrival_time_PcS"].values[0]
    arrival_scp  = row_match["arrival_time_ScP"].values[0]

    # P파 도착 시간을 상대 시간으로 변경
    p_absolute_time = event_time + arrival_p
    shift_in_seconds = -p_absolute_time.timestamp  # P파에 맞춰 0초

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
    tr_aligned.stats.pcsarr = shift_arrival_if_valid(arrival_pcs)
    tr_aligned.stats.scparr = shift_arrival_if_valid(arrival_scp)

    aligned_stream += tr_aligned

print(f"정렬 후, {len(aligned_stream)}개의 Trace를 섹션 플롯 합니다.")

###############################################################################
# 8.2) Matplotlib으로 파형 섹션 플롯 (P=0 기준)
###############################################################################
fig, ax = plt.subplots(figsize=(10, 12))

tmin, tmax = -100, 1000

ax.axvline(0, color="red", linestyle=":", linewidth=1.5, label="P")

# y축: 각거리
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
    data_plot *= 0.2

    dist_deg = getattr(tr.stats, "distance", np.nan)

    ax.plot(t_plot, dist_deg + data_plot, color="k", linewidth=0.8)

    line_half_height = 0.1

    # PcP(파란색)
    if getattr(tr.stats, "pcparr", None) is not None:
        xval = tr.stats.pcparr
        y1, y2 = dist_deg - line_half_height, dist_deg + line_half_height
        ax.plot([xval, xval], [y1, y2],
                color="blue", linestyle=":", linewidth=1.2)

    # S(초록색)
    if getattr(tr.stats, "sarr", None) is not None:
        xval = tr.stats.sarr
        y1, y2 = dist_deg - line_half_height, dist_deg + line_half_height
        ax.plot([xval, xval], [y1, y2],
                color="green", linestyle="--", linewidth=1.2)

    # ScS(보라색)
    if getattr(tr.stats, "scsarr", None) is not None:
        xval = tr.stats.scsarr
        y1, y2 = dist_deg - line_half_height, dist_deg + line_half_height
        ax.plot([xval, xval], [y1, y2],
                color="magenta", linestyle="--", linewidth=1.2)

    # PcS(주황색) - 필요하면 추가
    if getattr(tr.stats, "pcsarr", None) is not None:
        xval = tr.stats.pcsarr
        y1, y2 = dist_deg - line_half_height, dist_deg + line_half_height
        ax.plot([xval, xval], [y1, y2],
                color="orange", linestyle="--", linewidth=1.2)

    # ScP(검은색) - 필요하면 추가
    if getattr(tr.stats, "scparr", None) is not None:
        xval = tr.stats.scparr
        y1, y2 = dist_deg - line_half_height, dist_deg + line_half_height
        ax.plot([xval, xval], [y1, y2],
                color="black", linestyle="--", linewidth=1.2)

    # 스테이션 라벨(y축 오른쪽)
    ax.text(tmax + 50, dist_deg, f"{tr.stats.station}", va="center", fontsize=8)

# 플롯 범위
ax.set_xlim(tmin, tmax)
dist_vals = [tr.stats.distance for tr in aligned_stream if hasattr(tr.stats, "distance")]
if len(dist_vals) > 0:
    ymin, ymax = min(dist_vals), max(dist_vals)
else:
    ymin, ymax = 0, len(aligned_stream)
ax.set_ylim(ymin - 1, ymax + 1)

ax.set_xlabel("Time (s) relative to P arrival = 0")
ax.set_ylabel("Distance (deg)")
plt.title("Section Plot, aligned at P=0 \n(PcP=blue, S=green, ScS=magenta, PcS=orange, ScP=black)")
plt.tight_layout()
plt.show()