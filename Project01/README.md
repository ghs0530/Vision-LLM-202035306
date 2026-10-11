Project01 - Python 및 NumPy를 활용한 학생 성적 분석


1. 프로젝트 구현 내용


1.1 학생 이름 추출

학생 학번과 이름이 결합된 문자열에서 이름만 추출했습니다.

List Comprehension과 split() 함수를 활용했습니다.

student_names = [student.split("_")[1] for student in raw_students]


1.2 전체 성적 데이터 출력

NumPy의 2차원 배열에 저장된 학생별 성적을 1차원 배열로 변환했습니다.

flatten() 함수를 사용하여 총 15개의 성적을 출력했습니다.

all_scores = raw_scores.flatten()


1.3 85점 이상 성적 개수 계산

NumPy의 조건 연산과 sum() 함수를 활용했습니다.

전체 15개의 성적 중 85점 이상인 성적의 개수를 계산했습니다.

over_85 = (all_scores >= 85).sum()



1.4 학생별 평균 점수 계산

NumPy의 mean() 함수를 사용했습니다.

axis=1을 지정하여 각 학생의 세 과목 평균 점수를 계산했습니다.

zip()을 사용하여 학생 이름과 평균 점수를 함께 출력했습니다.

student_means = np.mean(raw_scores, axis=1)



1.5 최고 평균 학생 선정

argmax() 함수를 사용하여 평균 점수가 가장 높은 학생의 인덱스를 찾았습니다.

해당 인덱스를 이용해 학생의 이름과 평균 점수를 출력했습니다.

top_student_idx = np.argmax(student_means)
top_student_name = student_names[top_student_idx]



1.6 단일 과목 최고 점수 분석

max() 함수로 전체 성적 중 최댓값을 구했습니다.

argmax()로 최고 점수의 위치를 확인했습니다.

나머지 연산자 %와 정수 나눗셈 //를 활용하여 해당 과목과 학생을 구했습니다.

overall_max_score = np.max(raw_scores)
overall_max_idx = np.argmax(raw_scores)
overall_max_subject = subjects[overall_max_idx % len(subjects)]
overall_max_student = student_names[overall_max_idx // len(subjects)]



2. 실행 환경 및 의존성

사용 언어: Python 3

개발 환경: Visual Studio Code (VS Code)

제출 파일 형식: .py

사용 라이브러리: NumPy

필요한 라이브러리 설치

Python 3이 설치된 환경에서 다음 명령어를 실행합니다.

python -m pip install numpy



3. 실행 방법

Python 3을 설치합니다.

Visual Studio Code에서 프로젝트 폴더를 엽니다.

터미널에서 NumPy 라이브러리를 설치합니다.

Project01 폴더에 있는 project01.py 파일을 확인합니다.

프로젝트 최상위 폴더에서 다음 명령어를 실행합니다.

python Project01/project01.py

터미널에 출력된 학생별 성적 분석 결과를 확인합니다.




4. 실행 결과

1. 학생 명단: 짱구, 철수, 맹구, 유리, 훈이

2. 전산 시스템에 입력된 모든 성적: 95점, 82점, 88점, 70점, 65점, 90점, 98점, 100점, 92점, 60점, 75점, 80점, 81점, 88점, 84점

3. 전체 15개 성적 중 85점 이상 개수: 7개

4. 학생별 평균 점수: 짱구 88.33점, 철수 75.00점, 맹구 96.67점, 유리 71.67점, 훈이 84.33점

5. 최고 평균 수석 학생: 맹구 (96.67점)

6. 단일 과목 최고 점수: 100점 (과목: 수학, 학생: 맹구)






5. 참고 사항

학생 정보는 학번과 이름이 결합된 문자열 리스트로 구성했습니다.

성적 데이터는 5행 3열의 NumPy 배열로 구성했습니다.

각 행은 학생을, 각 열은 국어, 수학, 과학 과목을 나타냅니다.

평균 점수는 소수점 둘째 자리까지 표시했습니다.