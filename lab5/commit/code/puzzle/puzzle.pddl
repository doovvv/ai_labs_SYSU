(define (domain puzzle)
  (:requirements :strips :equality:typing)
  (:types num loc) 
  (:predicates
    (adjacent ?x - loc ?y - loc)
    (at ?x - num ?y - loc)
    )
  (:constants
    T0 - num
  )
  (:action slide
             :parameters (?t - num ?x ?y - loc)
             :precondition (and(at ?t ?x) (at T0 ?y) (adjacent ?x ?y))
             :effect (and (at ?t ?y) (at T0 ?x) (not (at ?t ?x)) (not (at T0 ?y))) 
  )
)