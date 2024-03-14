import pgmpy
import numpy as np
import pandas as pd
import math

class Analyse:
    def __init__(self):
        pass

    '''输入一个节点的MTBF,MTTR,t,返回第二个时间片的cpd'''
    @classmethod
    def Trans_cpd(cls,MTBF,MTTR,t):
        """
        Parameters:
        - MTBF:mean time between failure,相邻两次故障之间的平均工作时间,平均失效间隔时间
        - MTTR:mean time to repair,平均维修时间
        - t:time between two slices,set by user

        Returns:
        - cpd (pgmpy.factors.discrete.TabularCPD):cpd of the second time slice.
        e.g.+-------------+-------------+-------------+
            | ('Z', 0)    | ('Z', 0)(0) | ('Z', 0)(1) |
            +-------------+-------------+-------------+
            | ('Z', 1)(0) | 0.4         | 0.7         |
            +-------------+-------------+-------------+
            | ('Z', 1)(1) | 0.6         | 0.3         |
            +-------------+-------------+-------------+
        """
        try:
            if MTBF <= 0 or MTTR < 0 or t < 0:
                raise ValueError("Invalid parameter.")
        except ValueError as e:
            print(e)
            return None       

        node_cpd0 = np.zeros(2)
        node_cpd1 = np.ones(2)
        cpd = []
        lamda = 1 / MTBF
        P1 = 1 - math.e ** (- lamda * t)
        '''未给定MTTR时,认为MTTR值为0'''
        if MTTR == 0:
            node_cpd0[0] = 1 - P1
            node_cpd1[0] = P1
            cpd.append(node_cpd0)
            cpd.append(node_cpd1)
            return cpd
        
        mu = 1 / MTTR
        P2 = 1 - math.e ** (- mu * t)
        node_cpd0[0] = 1 - P1
        node_cpd0[1] = P2
        node_cpd1[0] = P1
        node_cpd1[1] = 1 - P2
        cpd.append(node_cpd0)
        cpd.append(node_cpd1)
        return cpd
    
    '''根据父节点的属性和逻辑门类型,返回子节点的cpd'''
    @classmethod
    def Logic_gate(cls, parent_name,parent_state,child_state,logic_gate):
        """
        Calculate the CPD of a child node given the parents' names and states, and the type oflogical gate
        operating between them.

        Parameters:
        - parent_name (str_list): Name of each parent node.
        - parent_state (int_list): State of each parent node.Each default by 0 or 1.
        - child_state (int): Number of the state of the child node.Default by 2.
        - logic_gate (str): The logical operation to perform on the parent node, ['AND', 'OR', 'NtoR', 'SP'].

        Returns:
        - child_cpd (pgmpy.factors.discrete.DiscreteCPD): The CPD of the child node.
        """
        try:
            if logic_gate not in ['AND', 'OR', 'NtoR', 'SP']:
                raise ValueError("Invalid logic gate type.")
            if len(parent_name)!= len(parent_state):
                raise ValueError("The number of parent names and states do not match.")
            if child_state != 2: 
                raise ValueError("Invalid child state.")

        except ValueError as e:
            print(e)
            return None
        else:
            print("Calculating CPD of child node...")
        """
        The storage of CPD is similiar to binary numbers.
        e.g.,the parents are 'A' and 'B',the logic gate is 'OR'.
        Then the probablity of the child is storaged as below.
        +----------+------+------+------+------+
        |    A     | A(0) | A(0) | A(1) | A(1) |
        +----------+------+------+------+------+
        |    B     | B(0) | B(1) | B(0) | B(1) |
        +----------+------+------+------+------+
        | child(0) | 1.0  | 0.0  | 0.0  | 0.0  |
        +----------+------+------+------+------+
        | child(1) | 0.0  | 1.0  | 1.0  | 1.0  |
        +----------+------+------+------+------+
        """
        if logic_gate == 'AND':
            '''
            所有父节点状态数的乘积,对应着子节点cpd的列数,命名为Num_child_prob.
            子节点默认只有0和1两种状态,所以子节点cpd只有两行
            '''
            Num_child_prob = math.prod(parent_state)
            child_0_prob = np.ones(Num_child_prob)
            child_0_prob[-1] = 0
            child_1_prob = np.zeros(Num_child_prob)
            child_1_prob[-1] = 1
            child_cpd = []
            child_cpd.append(child_0_prob)
            child_cpd.append(child_1_prob)
        
        if logic_gate == 'OR':
            Num_child_prob = math.prod(parent_state)
            child_0_prob = np.zeros(Num_child_prob)
            child_0_prob[0] = 1
            child_1_prob = np.ones(Num_child_prob)
            child_1_prob[0] = 0
            child_cpd = []
            child_cpd.append(child_0_prob)
            child_cpd.append(child_1_prob)

        if logic_gate == 'NtoR':
            R = int(input("Please input R:"))
            try:
                if R > len(parent_name) or R <= 0:
                    raise ValueError("Invalid R.")
            except ValueError as e:
                print(e)
                return None
                     
            Num_child_prob = math.prod(parent_state)
            child_0_prob = np.zeros(Num_child_prob)
            child_1_prob = np.zeros(Num_child_prob)
            child_cpd = []
            # Create a sudo binary array
            BiArray = np.zeros(len(parent_name))
            child_0_prob[0] = 1
            child_1_prob[0] = 0
            for i in range(1,2 ** len(parent_name)):
                BiArray[0] += 1
                for j in range(len(parent_name) - 1):
                    if BiArray[j] > 1:
                        BiArray[j] = 0
                        BiArray[j+1] += 1
                count = 0
                for k in range(len(parent_name)):
                    if BiArray[k] == 1:
                        count += 1
                if count >= R:
                    child_0_prob[i] = 0
                    child_1_prob[i] = 1
                else:
                    child_0_prob[i] = 1
                    child_1_prob[i] = 0
            child_cpd.append(child_0_prob)
            child_cpd.append(child_1_prob)

        if logic_gate == 'SP':
            # 输入父节点部件的MTBF和MTTR数组，时间间隔t，休眠因子α
            # 对于输入的MTBF和MTTR，先按照逗号分割，然后将字符串转换为浮点数
            MTBF_str = input("Please input MTBF,seperate by ',':")
            MTTR_str = input("Please input MTTR,seperate by ',':")
            MTBF = MTBF_str.strip(',').split(',')
            MTTR = MTTR_str.strip(',').split(',')
            MTBF = [float(x) for x in MTBF]
            MTTR = [float(x) for x in MTTR]
            t = float(input("Please input time interval:"))
            alpha = float(input("Please input diapause factor '\u03B1':"))
            try:
                if len(MTBF) != len(parent_name) or len(MTTR) != len(parent_name) or t <= 0:
                    raise ValueError("Invalid parameter.")
            except ValueError as e:
                print(e)
                return None
                
            lamda = 1 / np.array(MTBF)
            mu = 1 / np.array(MTTR)

            Num_child_prob = math.prod(parent_state)
            child_0_prob = np.zeros(Num_child_prob)
            child_1_prob = np.zeros(Num_child_prob)
            child_cpd = []
            # Create a sudo binary array
            BiArray = np.zeros(len(parent_name))
            BiArray[0] = -1 # 初始化，方便后续模拟二进制进位
            # NextP用来存储第二个时间片中,A1(t+1),…,An(t+1)取1的概率
            NextP = np.zeros(len(parent_name))

            '''定义一个函数,判断比部件Ai优先级更高的部件是否全部失效(值为1),如果是,则返回True,否则返回False'''
            def is_all_failure(A_list,Init_i):
                for i in range(Init_i):
                    if A_list[i] == 0 or A_list[Init_i] != 0:
                        return False
                return True
        
            for i in range(2 ** len(parent_name)):
                BiArray[0] += 1
                for j in range(len(parent_name) - 1):
                    if BiArray[j] > 1:
                        BiArray[j] = 0
                        BiArray[j+1] += 1
                if BiArray[0] == 0:
                    NextP[0] = 1 - math.e ** (- lamda[0] * t)
                    # A1=0时A1(t+1)=1的概率为1-e^(-λ*t)
                if BiArray[0] == 1:
                    NextP[0] = math.e ** (- mu[0] * t)
                    # A1=1时A1(t+1)=1的概率为e^(-μ*t)
                for Init_i in range(1,len(parent_name)):
                    if BiArray[Init_i] == 1:
                        NextP[Init_i] = math.e ** (- mu[Init_i] * t)
                        # Ai=1时Ai(t+1)=1的概率为e^(-μ*t)
                    elif is_all_failure(BiArray,Init_i):
                        NextP[Init_i] = 1 - math.e ** (- lamda[Init_i] * t)
                        # Ai=0时,如果比Ai优先级更高的部件全部失效,则A(t+1)=1的概率为1-e^(-λ*t)
                    else:
                        NextP[Init_i] = 1 - math.e ** (- alpha * lamda[Init_i] * t)
                
                # Mutiply represents the probability of the child node being 1,namely A1(t+1),…,An(t+1) all being 1.
                Mutiply = math.prod(NextP)
                child_0_prob[i] = 1 - Mutiply
                child_1_prob[i] = Mutiply
            child_cpd.append(child_0_prob)
            child_cpd.append(child_1_prob)

        return child_cpd