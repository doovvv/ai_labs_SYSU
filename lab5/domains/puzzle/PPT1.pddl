(define (problem PPT1)
 (:domain puzzle)
 (:objects 
    T1 T2 T3 T4 T5 T6 T7 T8 T9 T10 T11 T12 T13 T14 T15 - num
    P1 P2 P3 P4 P5 P6 P7 P8 P9 P10 P11 P12 P13 P14 P15 P16 - loc
 )
 (:init
    (at T14 P1) (at T10 P2) (at T6 P3) (at T0 P4) (at T4 P5) (at T9 P6) (at T1 P7) (at T8 P8)
    (at T2 P9) (at T3 P10) (at T5 P11) (at T11 P12) (at T12 P13) (at T13 P14) (at T7 P15) (at T15 P16)
    (adjacent P1 P2) (adjacent P1 P5)
    (adjacent P2 P1) (adjacent P2 P3) (adjacent P2 P6)
    (adjacent P3 P2) (adjacent P3 P7) (adjacent P3 P4)
    (adjacent P4 P3) (adjacent P4 P8)
    (adjacent P5 P6) (adjacent P5 P1) (adjacent P5 P9)
    (adjacent P6 P5) (adjacent P6 P10) (adjacent P6 P7) (adjacent P6 P2)
    (adjacent P7 P6) (adjacent P7 P11) (adjacent P7 P8) (adjacent P7 P3)
    (adjacent P8 P7) (adjacent P8 P12) (adjacent P8 P4)
    (adjacent P9 P13) (adjacent P9 P10) (adjacent P9 P5) 
    (adjacent P10 P9) (adjacent P10 P14) (adjacent P10 P11) (adjacent P10 P6)
    (adjacent P11 P10) (adjacent P11 P15) (adjacent P11 P12) (adjacent P11 P7)
    (adjacent P12 P11) (adjacent P12 P16) (adjacent P12 P8)
    (adjacent P13 P14) (adjacent P13 P9) 
    (adjacent P14 P13) (adjacent P14 P15) (adjacent P14 P10)
    (adjacent P15 P14) (adjacent P15 P16) (adjacent P15 P11)
    (adjacent P16 P15) (adjacent P16 P12)
    )
 
 (:goal 
    (and(at T1 P1) (at T2 P2) (at T3 P3) (at T4 P4) (at T5 P5) (at T6 P6) (at T7 P7) (at T8 P8) (at T9 P9)
    (at T10 P10) (at T11 P11) (at T12 P12) (at T13 P13) (at T14 P14) (at T15 P15))
 )
)
; 14 10 6 0
; 4 9 1 8
; 2 3 5 11
; 12 13 7 15