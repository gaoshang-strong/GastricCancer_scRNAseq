import pandas as pd

DE_list = pd.read_csv("/ShangGaoAIProjects/Zang/single_cell_data/data_analysis_pipeline_simplified/05_trajectory_of_T_cell/results_DE_in_effector_Resp_vs_nonResp/DE_effector_Responder_vs_NonResponder_wilcoxon.csv")

top50 = DE_list.head(50)
tail50 = DE_list.tail(50)

top50.to_csv("/ShangGaoAIProjects/Zang/single_cell_data/data_analysis_pipeline_simplified/05_trajectory_of_T_cell/results_DE_in_effector_Resp_vs_nonResp/Respondertop50_DE_genes.csv", index=False)
tail50.to_csv("/ShangGaoAIProjects/Zang/single_cell_data/data_analysis_pipeline_simplified/05_trajectory_of_T_cell/results_DE_in_effector_Resp_vs_nonResp/NonResponder_top50_DE_genes.csv", index=False)