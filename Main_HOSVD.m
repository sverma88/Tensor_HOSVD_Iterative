% Main Script to execute iterative tensor decomposition 
clc
Tensor_A_Train=Tensor_Data(1,1:8,:,:,:);
Tensor_A_Test=Tensor_Data(1,9:10,:,:,:);
Rank_A=[5,30,5,5];
Max_iterations=50;
Error_Threshold=10^-5;

% Decomposing Tensor (finding low rank Projection)
[Core_Tensor_A,Singular_Factors_TensorA,Original_Singular_Factors_TensorA]=Decompose_Tensor_Coupled_HOSVD_iteratively(Tensor_A_Train,Rank_A,Error_Threshold,Max_iterations);

% Feature Extraction in Low rank
Row_Projection=Singular_Factors_TensorA{3,1};
Col_Projection=Singular_Factors_TensorA{4,1};
[Projected_Images_Train_TensorA]=Project_Images(Tensor_A_Train,Row_Projection,Col_Projection);
[Projected_Images_Test_TensorA]=Project_Images(Tensor_A_Test,Row_Projection,Col_Projection);

save('ETH80_TensorA_Rank_5_30_5_5.mat','Singular_Factors_TensorA','Original_Singular_Factors_TensorA','Projected_Images_Train_TensorA','Projected_Images_Test_TensorA');