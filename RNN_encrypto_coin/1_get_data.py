#!/usr/bin/env python
# coding: utf-8

# https://apidocs.bithumb.com/reference/%EB%B6%84minute-%EC%BA%94%EB%93%A4-1
# https://api.bithumb.com/v1/candles/minutes/{unit}
# 빗썸에서 데이터 수신 : minutes, days, weeks, months
# order_currency 주문 통화, payment_currency 결제 통화, chart_intervals 차트 간격
# 날짜 포맷 : yyyy-MM-dd HH:mm:ss

def get_conindata(coinname="BTC", getunit="days", timm="", to=""):
    if getunit=="minutes" and not timm:
        timm = 60    
    # 오늘부터 전달받은 to 날짜까지(여기선 200 단위이므로 초과될 수 있음) loop 반영
    datasets = []
    conv_date = datetime.now()
    cut_date = datetime.strptime(to, "%Y-%m-%d %H:%M:%S")    
    
    if to :         
        while cut_date < conv_date :            
            to = conv_date.strftime("%Y-%m-%d %H:%M:%S")
            
            url = f"https://api.bithumb.com/v1/candles/{('minutes/'+str(timm) if getunit=='minutes' else getunit)}?market=KRW-{coinname}&count=200{'&to='+to if to else ''}"
            headers = {"accept": "application/json"}    
            response = requests.get(url, headers=headers)    

            #날짜 중복 체크 : datasets 가 담긴 이후 실행되므로 2번 loop부터 실행됨
            jData = json.loads(response.text)
            if datasets:
                higdDate = datetime.strptime(datasets[-1]["candle_date_time_kst"], "%Y-%m-%dT%H:%M:%S")
                lowDate = datetime.strptime(jData[0]["candle_date_time_kst"], "%Y-%m-%dT%H:%M:%S")
                if higdDate <= lowDate:
                    #중복 값 발생시
                    for ix in range(len(jData)):
                        lowDate = datetime.strptime(jData[ix]["candle_date_time_kst"], "%Y-%m-%dT%H:%M:%S")
                    if highDate < lowDate:
                        #datasets에 담기 전에 jData에서 해당 데이터 제거
                        jData.pop(ix)
                    else:
                        break
            
            datasets.extend(jData)
            conv_date = conv_date + dt.timedelta(days=-200)
    else :
        url = f"https://api.bithumb.com/v1/candles/{('minutes/'+str(timm) if getunit=='minutes' else getunit)}?market=KRW-{coinname}&count=200"
        headers = {"accept": "application/json"}    
        response = requests.get(url, headers=headers)
        datasets.extend(json.loads(response.text))  
    return datasets

# RNN 데이터 생성 함수
def create_rnn_data(datasets, timestep=30): #use_data = "pred"
    print(len(datasets))    
    #내림차순으로 된 데이터를 오름차순으로 정렬
    datasets.reverse()
    #timestep만큼 데이터 분할
    x_data = [];
    y_data = [];
    for ix in range(len(datasets) - timestep):
        #문제데이터 x : timestep일만큼의 데이터 timestep 앞자리까지
        #["opening_price", "high_price", "low_price", "trade_price", "prev_closing_price"] 필요데이터
        slice_data = [
            [d["opening_price"], d["high_price"], d["low_price"], d["trade_price"], d["prev_closing_price"]]
            for d in datasets[ix:timestep+ix]
        ]
        #x_data.append(datasets[ix:timestep+ix])
        x_data.append(slice_data)
        y_data.append(datasets[timestep+ix]) #정답데이터 y : 위 x 추세가 끝나고 그 다음날 가격데이터
    #y_data는 정답 1개만 반영 : 시작가와 현재가의 평균                    
    #y_data = [int((d["trade_price"])*10000)/10000 for d in y_data]
    y_data = [int((d["opening_price"]+d["trade_price"])/2*10000)/10000 for d in y_data]
    return np.array(x_data), np.array(y_data)

# 최고가데이터 가져오기
def high_create_rnn_data(datasets, timestep=30): #use_data = "pred"
    print(len(datasets))    
    #내림차순으로 된 데이터를 오름차순으로 정렬
    datasets.reverse()
    #timestep만큼 데이터 분할
    x_data = [];
    y_data = [];
    for ix in range(len(datasets) - timestep):
        #문제데이터 x : timestep일만큼의 데이터 timestep 앞자리까지
        #["opening_price", "high_price", "low_price", "trade_price", "prev_closing_price"] 필요데이터
        slice_data = [[d["high_price"]] for d in datasets[ix:timestep+ix]]
        #x_data.append(datasets[ix:timestep+ix])
        x_data.append(slice_data)
        y_data.append(datasets[timestep+ix]) #정답데이터 y : 위 x 추세가 끝나고 그 다음날 가격데이터
    #y_data는 정답 1개만 반영 : 시작가와 현재가의 평균                    
    #y_data = [int((d["trade_price"])*10000)/10000 for d in y_data]
    y_data = [ d["high_price"] for d in y_data ]
    return np.array(x_data), np.array(y_data)

# 30일 데이터 수신 함수
def get_xpred(coinname="BTC", getunit="days", timm="", timestep=30):    
    url = f"https://api.bithumb.com/v1/candles/{('minutes/'+str(timm) if getunit=='minutes' else getunit)}?market=KRW-{coinname}&count={timestep+1}"
    headers = {"accept": "application/json"}    
    response = requests.get(url, headers=headers)
    datasets = json.loads(response.text)    
    if len(datasets) < timestep + 1:
        return np.array([])
    datasets.reverse()
    #cur_price = datasets[-1]["trade_price"]
    y_cur_price = int((datasets[-1]["opening_price"]+datasets[-1]["trade_price"])/2*10000)/10000
    slice_data = [
        [d["opening_price"], d["high_price"], d["low_price"], d["trade_price"], d["prev_closing_price"]]
        for d in datasets[-timestep:]
    ]
    x_data = slice_data
    slice_data = [
        [d["opening_price"], d["high_price"], d["low_price"], d["trade_price"], d["prev_closing_price"]]
        for d in datasets[-timestep-1:-1]
    ]
    x_pre_data = slice_data
    return np.array(x_data), np.array(x_pre_data), y_cur_price

def get_high_xpred(coinname="BTC", getunit="days", timm="", timestep=30):    
    url = f"https://api.bithumb.com/v1/candles/{('minutes/'+str(timm) if getunit=='minutes' else getunit)}?market=KRW-{coinname}&count={timestep+1}"
    headers = {"accept": "application/json"}    
    response = requests.get(url, headers=headers)
    datasets = json.loads(response.text)    
    if len(datasets) < timestep + 1:
        return np.array([])
    datasets.reverse()
    #cur_price = datasets[-1]["trade_price"]
    y_cur_price = datasets[-1]["high_price"] * 1
    slice_data = [[d["high_price"]] for d in datasets[-timestep:]]
    x_data = slice_data
    slice_data = [[d["high_price"]] for d in datasets[-timestep-1:-1]]
    x_pre_data = slice_data
    return np.array(x_data), np.array(x_pre_data), y_cur_price

# 산점도 그래프 그려줄 함수
import matplotlib.pyplot as plt
def anal_scatter(x_data, y_data):
    print(x_data[0][0])
    #opening_price 시작가, high_price 최고가, low_price 최저가, trade_price 현재가, candle_acc_trade_price 거래액
    #candle_acc_trade_volume 거래량, prev_closing_price 전일 종가, change_price 변동액, change_rate 변동율
    anal_list = [
        "opening_price", "high_price", "low_price", "trade_price", "candle_acc_trade_price", 
        "candle_acc_trade_volume", "prev_closing_price", "change_price", "change_rate"
    ]
    plt.figure(figsize=(8,4))
    for i in range(9):
        plt.subplot(2,5,i+1)
        # 내부는 딕셔너리 구조이므로 변환
        x_anal = []
        for d in x_data:          
            x_anal.append([f[anal_list[i]] for f in d])            
        #plt.scatter(x_anal, y_data)
        plt.scatter(np.array(x_anal).mean(axis=1), y_data, s=1)
        plt.title(anal_list[i])
    plt.show()

if __name__ == "__main__":
    #res = get_conindata("BTC", to="2024-05-01%2000:00:00") #%20 공백처리
    rawdata = get_conindata("BTC", to="2016-03-02 00:00:00")    
    x_data, y_data = create_rnn_data(rawdata)
    print(y_data[0]) #현재가격
    #anal_scatter(x_data, y_data)
    #opening_price, high_price, low_price, trade_price, prev_closing_price 5개 항목만 연관성
    print(x_data[0][0])    
    print(y_data[0])    

# #datetime 타입
# current = datetime.datetime.now() #현재 datetime
# print(current, ":::", type(current))
# print(current.date()) #타임스탬프에서 yyyy-MM-dd
# print(current.time()) #타임스탬프에서 hh:mm:ss .마이크로 세컨(소수점 6자리)
# #timedelta 날짜의 차이 또는 시간 차
# mydate = datetime.datetime.strptime(f"{'2/29'};1984", "%m/%d;%Y") #문자 형태에 맞게 datetime 반환
# print(mydate)
# d = datetime.date(2005, 7, 14)
# t = datetime.time(12, 30)
# print(datetime.datetime.combine(d,t)) #날짜와 시간 결합
# dt = datetime.datetime.strptime("21일11월06년 16시30분", "%d일%m월%y년 %H시%M분") #문자 양식에 맞게 datetime으로 반환
# print(dt)

# #time 타입
# print(datetime.time.fromisoformat('04:23:01'))
# print(datetime.time.fromisoformat('04:23:01.003344'))

# #date 타입
# customdate = datetime.date(2025,1,25) #년월일 생성
# print(customdate)
# print(type(customdate))
# customdate = customdate.replace(day=26) #값 변경
# print(customdate)
# customdate2 = datetime.date.fromisoformat("2025-01-25") #str to date
# print(customdate2.isoformat()) #date to str
# print(customdate.strftime("%y년 %m월 %d일")) # date 타입의 날짜를 문자로 반환
# print(customdate.timetuple()) #각 값을 튜플 형태로 반환
# print(t.replace(hour=21)) #time 타입에서 값 변경

# #밀리세컨 : 1970.01.01 00:00:00 부터 현재까지 지난 시간을 1/1000초 단위로 반환
# ts = current.timestamp() #datetime to timestamp
# print(ts)
# print(datetime.datetime.fromtimestamp(ts)) #timestamp to datetime

# #현재 날짜에서 크리스마스까지 몇일, 몇시간, 몇분, 몇초가 남았는지 계산
# print("====================================")
# today = datetime.datetime.now()
# x_mas = datetime.datetime(2025,12,25, 0, 0, 0)
# diff = x_mas - today #timedelta 타입으로 반환됨
# print(diff)
# d_day = diff.days
# d_hour = diff.seconds // (60 * 60)
# re_sec = diff.seconds % (60 * 60)
# d_min = re_sec // 60
# d_sec = re_sec % 60

# print("크리스마스까지 [",d_day, "일", d_hour, "시간", d_min, "분", d_sec, "초 ] 남음")

