import requests
import jsons
import numpy as np

# 거래데이터 row 정보 가져오기
def getCandleData(ttime="days", cname="BTC"):
    get_addr = r"https://api.bithumb.com/v1/candles/{}?market=KRW-{}&count=200".format(ttime, cname)    
    response = requests.get(get_addr)    
    candle_data = jsons.load(response.json())    
    return candle_data

# 훈련데이터 만들기
def createX(dataset, transCount) :    
    dataset.reverse() 
    del dataset[-1]
    xlist = []
    ylist = []
    for i in range(len(dataset) - transCount):
        tmp = dataset[i:transCount+i]
        for data in tmp:            
            if "market" in data :
                del data["market"];
            if "candle_date_time_utc" in data :
                del data["candle_date_time_utc"];
            if "timestamp" in data :
                del data["timestamp"];
            if "change_price" in data :
                del data["change_price"];                
            if "change_rate" in data :
                del data["change_rate"];
            if "candle_date_time_kst" in data :
                del data["candle_date_time_kst"];
            if "prev_closing_price" in data : 
                del data["prev_closing_price"];
            if "first_day_of_period" in data : 
                del data["first_day_of_period"];
        xlist.append(tmp)
        ylist.append(
            [
                dataset[transCount+i]["opening_price"],                
                dataset[transCount+i]["high_price"],
                dataset[transCount+i]["low_price"],
                dataset[transCount+i]["trade_price"]
            ]                
        )
    return xlist, np.array(ylist)

# 딕셔너리 구조에서 value 만 추출한뒤 numpy로 변형하여 리턴
def integeration_xdata(xdata) :
    keylist = xdata[0][0].keys()    
    xlist = [] 
    for d in xdata :
        tmp = []
        for f in d :
            tmp.append(list(f.values()))
        xlist.append(tmp)    
    return np.array(xlist), keylist