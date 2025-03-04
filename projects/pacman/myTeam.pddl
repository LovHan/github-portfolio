;Header and description

(define (domain pacman_bool)

    ;remove requirements that are not needed
    (:requirements :strips :typing)
    (:types 
        enemy team - object
        enemy1 enemy2 - enemy
        ally current_agent - team
    )

    ; un-comment following line if constants are needed
    ;(:constants )
    ;(:types food)
    (:predicates 

        ;Basic predicates
        (enemy_around ?e - enemy ?a - team) ;enemy ?e is within 4 grid distance with agent ?a
        (is_pacman ?x) ; if an agent is pacman 
        (food_in_backpack ?a - team)  ; have food in backpack
        (food_available) ; still have food on enemy land

        ;Predicates for virtual state to set goal states
        (defend_foods) ;The environment do not collect state for this predicates, this is a virtual effect state for action patrol 


        ;Advanced predicates
        ;These predicates are currently not used and consider the state of other agent 
        (enemy_long_distance ?e - enemy ?a - current_agent) ; noisy distance return longer than 25 
        (enemy_medium_distance ?e - enemy ?a - current_agent) ; noisy distance return longer than 15 
        (enemy_short_distance ?e - enemy ?a - current_agent) ; noisy distance return shorter than 15 

        (3_food_in_backpack ?a - team) ; more than 3 food in backpack
        (5_food_in_backpack ?a - team)  ; more than 5 food in backpack
        (7_food_in_backpack ?a - team)  ; more than 7 food in backpack
        (13_food_in_backpack ?a - team) ; more than 12 food in backpack
        (17_food_in_backpack ?a - team) ; more than 12 food in backpack
        (15_food_in_backpack ?a - team) ; more than 12 food in backpack
        (10_food_in_backpack ?a - team)    ; more than 10 food in backpack
        (20_food_in_backpack ?a - team)    ; more than 20 food in backpack

        (near_food ?a - current_agent)  ; a food within 4 grid distance 
        (near_capsule ?a - current_agent)   ;a capsule within 4 grid distance
        (capsule_available) ; capsule available on map
        (winning)   ; is the team score more than enemy team
        (winning_gt3) ; is the team score 3 more than enemy team
        (winning_gt5)    ; is the team score 5 more than enemy team
        (winning_gt10)  ; is the team score 10 more than enemy team
        (winning_gt20)  ; is the team score 20 more than enemy team
        (near_ally  ?a - current_agent) ; is ally near 4 grid distance
        (is_scared ?x) ;is enemy, current agent, or the ally in panic (due to capsule eaten by other side)


        ;Cooperative predicates
        ;The states of the following predicates are not collected by demo team_ddl code;
        ;To use these predicates, You have to collect the corresponding states when preparing pddl states in the code.
        ;These predicates describe the current action of your ally
        (eat_enemy ?a - ally)
        (go_home ?a - ally)
        (go_enemy_land ?a - ally)
        (eat_capsule ?a - ally)
        (eat_food ?a - ally)

    )

    ;define actions here
        
    (:action attack
        :parameters (?a - current_agent ?e1 - enemy1 ?e2 - enemy2 )
        :precondition (and 
            (food_available)  )
        :effect (and 
            (not (food_available))
            (not (is_pacman ?e1))  ; Add this effect - attacking also clears enemy pacmen
            (not (is_pacman ?e2))
        )
    )

    ; Escape and score action
    ; Simpliest strategy of run home if have food
    (:action escape_and_score
        :parameters (?a - current_agent)
        :precondition (and 
            (is_pacman ?a)
            (food_in_backpack ?a)
        )
        :effect (and 
            (not (is_pacman ?a))
        )
    )

    (:action defence
        :parameters (?a - current_agent ?e - enemy)
        :precondition (and 
            (is_pacman ?e)
            (not (is_pacman ?a))
        )
        :effect (and 
            (not (is_pacman ?e))
        )
    )

    (:action collect_nearby_food
        :parameters (?a - current_agent)
        :precondition (and 
            (is_pacman ?a)
            (food_in_backpack ?a)
            (near_food ?a)
            (not (15_food_in_backpack ?a))  
        )
        :effect (and 
            (15_food_in_backpack ?a)  
        )
    )

    (:action go_home
        :parameters (?a - current_agent)
        :precondition (and 
            (is_pacman ?a)
        )
        :effect (and 
            (not (is_pacman ?a))
        )
    )

    (:action patrol
        :parameters (?a - current_agent ?e1 - enemy1 ?e2 - enemy2)
        :precondition (and 
            (not (is_pacman ?a))
            (winning_gt10)
        )
        :effect (and 
            (defend_foods)
        )
    )

    (:action retreat_to_safety
        :parameters (?a - current_agent ?e - enemy)
        :precondition (and 
            (is_pacman ?a)
            (enemy_around ?e ?a)
        )
        :effect (and 
            (not (is_pacman ?a))
        )
    )

    (:action eat_capsule
        :parameters (?a - current_agent ?ally - ally ?e1 - enemy1 ?e2 - enemy2)
        :precondition (and 
            (capsule_available)
            (near_capsule ?a)
            (is_pacman ?a)
        )
        :effect (and 
            (not (capsule_available))
            (is_scared ?e1)
            (is_scared ?e2)
            (eat_enemy ?a)        ; Current agent assigned to hunt enemies
            (eat_food ?ally)      ; Ally assigned to collect food
        )
    )
)
