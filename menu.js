menu_sets = []
//메뉴 생성기 시작 S==============================
class Menu{
    constructor(mtitle){
        this.mtitle=mtitle;
    }
    mtitle;url;tips;
}
//?menu=0 의 쿼리스트링은 데이터 생성함수 data_sets 객체의 push 순서와 일치합니다.
menu0 = new Menu("1. 선형회귀 모델 구축")//1. 개발목적/언어-선정/요구사항명세/분석 2. 구현도구/라이센스 3. 테스트도구(junit,mockobj)  4. 형상관리도구  5. 빌드도구
menu0.url = "?menu=0"
menu0.tips = ["1.1 보스턴 주택가격 예측","1.2 캘리포니아 주택가격 예측","1.3 1년 후 당뇨 상태 예측"]
menu1 = new Menu("2. 분류 모델 구축")
menu1.url = "?menu=1"
menu1.tips = ["2.1 fashion_mnist_softmax","2.2 CNN(컨볼루션 레이어 적용 모델)","2.3 가상화폐 가격분석 회귀(실무)"] 
menu2 = new Menu("3. 서버프로그램 구현")
menu2.url = "?menu=2"
menu2.tips = ["3.1 회원가입 구현","3.2 회원 로그인 구현","3.3 회원 로그아웃 구현","3.4 회원 리스트 출력 구현 ","3.5 테스트케이스","3.6 Mock Object 테스트","3.7 테스트 결과보고서"]
menu3 = new Menu("4. 배치프로그램구현")
menu3.url = "?menu=3"
menu3.tips = ["4.1 배치스케줄구성 및 테스트"]// (5분마다 프로그램 정상 작동유무 출력 메시지 스케줄링 확인)



menu_sets.push(menu0)
menu_sets.push(menu1)
menu_sets.push(menu2)
menu_sets.push(menu3)

