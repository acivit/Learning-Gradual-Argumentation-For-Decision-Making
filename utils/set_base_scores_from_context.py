from settings import IMPORTANCES_TEST

def set_base_scores_from_context(context,):

    elevator_non_stretched_capacity=7
    base_scores = {}

    if context["next_time_over_timeout"] == True:
        base_scores["next_time_over_timeout"] = 1
        #base_scores["next_time_not_over_timeout"] = 0
    else:
        base_scores["next_time_over_timeout"] = 0
        #base_scores["next_time_not_over_timeout"] = 1


    base_scores["percentage_timeout"] = min(1, float(context["robot_time_waiting"]) / float(context["timeout"]))

    base_scores["inside_comfort"] = 0.5
    base_scores["breaking_queue"] = 0.5
    base_scores["robot_urgency"] = 0.5
    base_scores["percentage_stretchment"] =  min(1, (context["number_people_in_elevator"] + len(context["users_before_robot"])+1) / float(context["elevator_capacity"])) #* cultural_stretchment_importance we skip this but could be added...
        
    users_before_robot = context["users_before_robot"]
    
    if users_before_robot:
        
        users_roles = [user["role"] for user in users_before_robot]
        number_users_before_robot = len(users_before_robot)
        base_scores["amount_users_queue"] = min(1, number_users_before_robot * IMPORTANCES_TEST[context["robot_profile"]]["each_user_in_queue"])
        if context["waiting_users_did_not_enter"] == True:
            base_scores["waiting_users_did_not_enter"] = 1
        else:
            base_scores["waiting_users_did_not_enter"] = 0


        if "priority" in users_roles:
            base_scores["important_role_in_queue"] = 1
        else:
            base_scores["important_role_in_queue"] = 0

    else:
        #base_scores["breaking_queue"] = 0
        base_scores["amount_users_queue"] = 0
        base_scores["waiting_users_did_not_enter"] = 0
        base_scores["important_role_in_queue"] = 0


    if context["people_with_impairments"] == True and (context["number_people_in_elevator"] + len(context["users_before_robot"]) + 1) > elevator_non_stretched_capacity:
        base_scores["people_impairment"] = 1
    else:
        base_scores["people_impairment"] = 0

    return base_scores




def set_base_scores_original(context):
    
    number_people_in_elevator = context["number_people_in_elevator"] if "number_people_in_elevator" in context else None
    elevator_capacity = context["elevator_capacity"] if "elevator_capacity" in context else None
    elevator_non_stretched_capacity = context["elevator_non_stretched_capacity"] if "elevator_non_stretched_capacity" in context else None
    cultural_stretchment_importance = context["cultural_stretchment_importance"] if "cultural_stretchment_importance" in context else None
    people_with_impairments = context["people_with_impairments"] if "people_with_impairments" in context else None
    robot_time_waiting = context["robot_time_waiting"] if "robot_time_waiting" in context else None
    next_elevator_time = context["next_elevator_time"] if "next_elevator_time" in context else None
    timeout = context["timeout"] if "timeout" in context else None
    users = context["users_before_robot"] if "users_before_robot" in context else []
    waiting_users_did_not_enter = context["waiting_users_did_not_enter"] if "waiting_users_did_not_enter" in context else None

    base_scores = {"robot_urgency": IMPORTANCES_TEST[context["robot_profile"]]["robot_urgency"]}
    elevator_non_stretched_capacity=7

    if people_with_impairments == True and (number_people_in_elevator + len(users) + 1) > elevator_non_stretched_capacity:
        base_scores["people_impairment"] = IMPORTANCES_TEST[context["robot_profile"]]["people_impairment"]
    else:
        base_scores["people_impairment"] = 0

    if context["next_time_over_timeout"] == True:
        base_scores["next_time_over_timeout"] = IMPORTANCES_TEST[context["robot_profile"]]["next_time_over_timeout"]
        base_scores["next_time_not_over_timeout"] = 0
    else:
        base_scores["next_time_not_over_timeout"] = IMPORTANCES_TEST[context["robot_profile"]]["still_time_for_timeout"]
        base_scores["next_time_over_timeout"] = 0

    base_scores["percentage_timeout"] = IMPORTANCES_TEST[context["robot_profile"]]["percentage_timeout"] * min(1, robot_time_waiting / timeout)
    
    # TODO: Think if task importance should play a role in the decisions. 
    #activatable_arguments["task_importance"] = IMPORTANCES_TEST["task_importance"] * (1 if task_importance == "high" else 0)

    users_before_robot = users
    users_roles = [user["role"] for user in users_before_robot]

    number_users_before_robot = len(users_before_robot)

    if number_people_in_elevator is not None:
        base_scores["inside_comfort"] = IMPORTANCES_TEST[context["robot_profile"]]["inside_comfort"]
        base_scores["percentage_stretchment"] = IMPORTANCES_TEST[context["robot_profile"]]["percentage_stretchment"] * min(1, (number_people_in_elevator + number_users_before_robot+1) / elevator_capacity) * 0.5# cultural_stretchment_importance
    
    if number_users_before_robot > 0:
        base_scores["breaking_queue"] = IMPORTANCES_TEST[context["robot_profile"]]["breaking_queue"]
        base_scores["amount_users_queue"] = min(1, number_users_before_robot * IMPORTANCES_TEST[context["robot_profile"]]["each_user_in_queue"])
        if waiting_users_did_not_enter == True:
            base_scores["waiting_users_did_not_enter"] = IMPORTANCES_TEST[context["robot_profile"]]["waiting_users_did_not_enter"]
        else:
            base_scores["waiting_users_did_not_enter"] = 0
        if len(users_roles)>0:
            if "priority" in users_roles:
                base_scores["important_role_in_queue"] = IMPORTANCES_TEST[context["robot_profile"]]["important_role_in_queue"]
            else:
                base_scores["important_role_in_queue"] = 0

    else:
        base_scores["breaking_queue"] = 0
        base_scores["amount_users_queue"] = 0
        base_scores["waiting_users_did_not_enter"] = 0
        base_scores["important_role_in_queue"] = 0

    return base_scores