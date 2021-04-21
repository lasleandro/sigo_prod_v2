import re


##############################
def FindEmail(input_string):
    
    try:
        emails_list = re.findall(r'([a-zA-Z0-9._-]+@[a-zA-Z0-9._-]+\.[a-zA-Z0-9_-]+)', input_string)
    except:
        emails_list = [' ']
        
    return emails_list[0]
###############################

#########################################
def FindNumbersBraket(input_string):
    
    try:
        numbers_list = re.findall(r"\[\d-\d+\]", input_string)
        
        numbers_list = [item.replace('[', '').replace(']', '') for item in numbers_list]
        
    except:
        numbers_list = ['']
        
        
    if len(numbers_list) == 0:
        numbers_list = ['']
        
    return numbers_list[0]
#########################################



############################################
def FindLawsuit(input_string):
    
    regex_expression = '\d+-\d+\.\d+\.\d+\.\d+\.\d+'
    regex_expression2 = '\d+-\d+\.\d+\.\d+\.\d+'
    regex_expression3 = '\d+\.\d+\.\d+\.\d+\-\d+'
    regex_expression4 = '\d+-\d+\.\d+\.\d+'
    regex_expression5 = '\d+/\d+\.\d+\.\d+-\d+'
    regex_expression6 = '\d+\.\d+\.\d+-\d+'
    
    
    try:
        numbers_list = re.findall(regex_expression, input_string)
        numbers_list2 = re.findall(regex_expression2, input_string)
        numbers_list3 = re.findall(regex_expression3, input_string)
        numbers_list4 = re.findall(regex_expression4, input_string)
        numbers_list5 = re.findall(regex_expression5, input_string)
        numbers_list6 = re.findall(regex_expression6, input_string)
        numbers_list = numbers_list + numbers_list2 + numbers_list3 + numbers_list4 +\
                       numbers_list5 + numbers_list6
        numbers_list = [item for item in numbers_list if item != '']
                
    except:
        numbers_list = ['']
        
        
    if len(numbers_list) == 0:
        numbers_list = ['']
        
    return numbers_list[0]
#################################################


##############################################
def FindCaseId(input_string):
    
    try:
        numbers_list = re.findall(r"\d-\d+", input_string)
        
        numbers_list = [item.replace('[', '').replace(']', '') for item in numbers_list]
        
    except:
        numbers_list = ['']
        
        
    if len(numbers_list) == 0:
        numbers_list = ['']
        
    return numbers_list[0]
####################################################