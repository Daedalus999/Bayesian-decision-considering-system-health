from Analyse import Analyse
from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import DynamicBayesianNetwork as DBN
from pgmpy.inference import DBNInference

if __name__ == '__main__':
    parent_name = ['A', 'B','C','D']
    parent_state = [2,2,2,2]
    child_cpd = Analyse.Logic_gate(parent_name,parent_state,2,'SP')
    cpd= TabularCPD(variable='child', variable_card=2, values=child_cpd,\
                    evidence=parent_name,evidence_card=parent_state)
    print(cpd)
    
    