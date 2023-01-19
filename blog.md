알아낸 사실
아까 prophet에서 add_regressor

 Regressor 'name' missing from dataframe가 떴다.

 해결:

 df에 해당 name의 column이 있어야 하고, future df에도 해당 name의 column이 있어야 한다.



 Facebook Prophet 라이브러리는 시계열 예측을 위한 도구로, 시계열 데이터에서 트렌드와 계절성을 자동으로 감지하여 예측합니다.

Prophet에서 forecast() 함수를 사용하면, 주어진 기간 동안의 예측값을 포함하는 DataFrame을 반환합니다. 그리고 결측값에 대한 예상값을 확인하려면 yhat 컬럼을 참조하면 됩니다.

그러나, Prophet은 이전에 정의되지 않은 결측값을 예측하는 기능을 기본적으로 제공하지 않습니다. 그러므로 결측값을 처리하기 전에 데이터를 전처리해야 할 수도 있습니다.