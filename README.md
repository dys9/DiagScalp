# Detection keratin in the scalp


**1) 각질 Detection 함수**
    
    createMask()    : 모발 마스크 이미지 생성               
    //  입력 = 파일이름                           
    //  출력 = hair_mask
    
    keratinDetect() : 각질 이미지 생성                     
    //  입력 = 파일이름, hair_mask, 계산 방식      
    //  출력 = 경계값, 흑백 각질 영상
    
    keratinDraw()   : 각질 원본영상에 표시                 
    //  입력 = 원본파일, 경계값                    
    //  출력 = 각질이 색칠된 영상

**2) TrackBar 함수**

    setTracker()    : 경계값 생성                        
    //  입력 = pos                                
    //  출력 = pos 따른, 경계값
    
    onMethod()      : MORPH_TOPHAT or MORPH_GRADIENT    
    // 입력 = pos                                 
    // 출력 = 변화량 탐색 방법에 따른 각질 영상
    
    onThresh()      : 경계값에 따른 각질 출력             
    // 입력 = pos                                 
    // 출력 = 경계값에 따른 각질 영상

**3) 작동방법**

    - Default를 1로 설정할 경우, 계산적 경계값만 사용한 각질 영상 출력
    - Method를 1로 설정할 경우 MORPH_GRADIENT 연산 사용, Method를 2로 설정할 경우 MORPH_TOPHAT 연산 사용
    - Sense 값을 높일 경우, 더 많은 각질 추정 범위 출력
