class CustomEncoder():    
    # 파이선에서 __ (언더바 2개)는 private(외부접근 제한)
    def __init__(self, mylist=None):
        if mylist :
            self.__mylist = sorted(list(set(mylist))) #중복데이터 제거
            self.__convInt = [ ix for ix, _ in enumerate(self.__mylist) ] #인덱스를 정수로변환 #enumerate : 인덱스와 값이 같이 반환됨            
        else :
            self.__mylist = None
            self.__convInt = None                      
        print(self.__convInt)
    #정수변환
    def label_to_integer(self, y_dataset):
        if self.__mylist:
            print("mylist를 먼저 입력해야 합니다.")
            return None
        return [ self.__mylist.index(d) for d in y_dataset ]
    #라벨 원핫 인코딩
    def label_to_one_hot(self, y_dataset):
        if self.__mylist:
            print("mylist를 먼저 입력해야 합니다.")
            return None
        tmp = y_dataset.copy() # 원본데이터 변경 안되게
        tmp = self.label_to_integer(tmp) #모두 정수로 변환
        print(tmp)
        rettmp = []
        retlist = [0 for i in range(len(self.__mylist))] #0으로 이루어진 배열
        for d in tmp:
            rtmp = retlist.copy() #0으로 이루어진 배열 copy
            rtmp[d] = 1 #해당 인덱스 값만 1로 변환
            rettmp.append(rtmp)
        return self.__mylist, tmp, rettmp #원본데이터 중복제거, 원본데이터 정수변환, 원핫인코딩
    #정수 원핫 인코딩
    def integer_to_one_hot(self, y_intdataset, labeldata=None):
        if labeldata:            
            self.__mylist = labeldata.copy()
            self.__convInt = [ ix for ix, _ in enumerate(self.__mylist) ]
        maxdata = max(y_intdataset)        
        retlist = [0 for i in range(maxdata+1)]
        rettmp = []
        for d in y_intdataset:
            rtmp = retlist.copy()
            rtmp[d] = 1
            rettmp.append(rtmp)
        return rettmp
    #원핫 인코딩 데이터를 정답으로 반환
    def one_hot_to_label(self, oharr, label_list=None):        
        import numpy as np
        if label_list:
            temp = [ label_list[np.argmax(ohdata)] for ohdata in oharr ]
        else:
            print(self.__mylist)
            temp = [ self.__mylist[np.argmax(ohdata)] for ohdata in oharr ]
        return temp
# 1. 생성자 이용시 : __init__
    #  - 라벨 파라미터 : 문자형 리스트 = 정답데이터
    #  - 라벨 파라미터 존재시 : 라벨을 저장하고, 정수형 변수를 저장해 놓는다.
    #  - 라벨 파라미터 없을시 : 라벨과 정수형 변수를 관리하지 않는다.
    # 2. label_to_integer(y_dataset) : 파라미터 구분기호 : [] 생략가능, () 필수
    #  - 라벨이 존재하는 경우 y_dataset(라벨데이터)를 정수형으로 반환
    #  - ret : integer list type
    # 3. label_ont_hot(y_dataset)
    #  - 라벨 존재시 라벨리스트를 받아 원핫인코딩으로 반환
    # 4. integer_to_one_hot(self, y_inedatasaet, labeldata=None)
    #  - 정수형 데이터를 받아 원핫 인코딩으로 반환
    # 5. one_hot_to_label(self, oharr, label_list=None)
    #  - 원핫인코딩된 데이터 및 라벨리스트를 받아 라벨데이터리스트로 반환
#해당 파일(페이지)에서만 실행
print(__name__)
if __name__=="__main__":
    pass