??
??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
;
Sub
x"T
y"T
z"T"
Ttype:
2	
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.4.12v2.4.0-49-g85c8b2a817f8??
x
layer-1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:	2*
shared_namelayer-1/kernel
q
"layer-1/kernel/Read/ReadVariableOpReadVariableOplayer-1/kernel*
_output_shapes

:	2*
dtype0
p
layer-1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*
shared_namelayer-1/bias
i
 layer-1/bias/Read/ReadVariableOpReadVariableOplayer-1/bias*
_output_shapes
:2*
dtype0
v
output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2*
shared_nameoutput/kernel
o
!output/kernel/Read/ReadVariableOpReadVariableOpoutput/kernel*
_output_shapes

:2*
dtype0
n
output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameoutput/bias
g
output/bias/Read/ReadVariableOpReadVariableOpoutput/bias*
_output_shapes
:*
dtype0
l
VariableVarHandleOp*
_output_shapes
: *
dtype0*
shape
:	*
shared_name
Variable
e
Variable/Read/ReadVariableOpReadVariableOpVariable*
_output_shapes

:	*
dtype0

NoOpNoOp
?
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?
value?B? B?
?
layer-0
layer_with_weights-0
layer-1
	variables
regularization_losses
trainable_variables
	keras_api

signatures
 
i

Layers
		model

	variables
regularization_losses
trainable_variables
	keras_api
#
0
1
2
3
4
 

0
1
2
3
?
layer_metrics
	variables
layer_regularization_losses
regularization_losses

layers
trainable_variables
non_trainable_variables
metrics
 

0
1
2
?
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
	variables
regularization_losses
trainable_variables
	keras_api
#
0
1
2
3
4
 

0
1
2
3
?
layer_metrics

	variables
 layer_regularization_losses
regularization_losses

!layers
trainable_variables
"non_trainable_variables
#metrics
JH
VARIABLE_VALUElayer-1/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE
HF
VARIABLE_VALUElayer-1/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUEoutput/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE
GE
VARIABLE_VALUEoutput/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE
DB
VARIABLE_VALUEVariable&variables/4/.ATTRIBUTES/VARIABLE_VALUE
 
 

0
1

0
 
Y
w
$	variables
%regularization_losses
&trainable_variables
'	keras_api
h

kernel
bias
(	variables
)regularization_losses
*trainable_variables
+	keras_api
h

kernel
bias
,	variables
-regularization_losses
.trainable_variables
/	keras_api
#
0
1
2
3
4
 

0
1
2
3
?
0layer_metrics
	variables
1layer_regularization_losses
regularization_losses

2layers
trainable_variables
3non_trainable_variables
4metrics
 
 

0
1
2
	3

0
 

0
 
 
?
5layer_metrics
$	variables
6layer_regularization_losses
%regularization_losses

7layers
&trainable_variables
8non_trainable_variables
9metrics

0
1
 

0
1
?
:layer_metrics
(	variables
;layer_regularization_losses
)regularization_losses

<layers
*trainable_variables
=non_trainable_variables
>metrics

0
1
 

0
1
?
?layer_metrics
,	variables
@layer_regularization_losses
-regularization_losses

Alayers
.trainable_variables
Bnon_trainable_variables
Cmetrics
 
 

0
1
2

0
 
 
 
 

0
 
 
 
 
 
 
 
 
 
 
 
z
serving_default_input_3Placeholder*'
_output_shapes
:?????????	*
dtype0*
shape:?????????	
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_3Variablelayer-1/kernellayer-1/biasoutput/kerneloutput/bias*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8? *.
f)R'
%__inference_signature_wrapper_1899621
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename"layer-1/kernel/Read/ReadVariableOp layer-1/bias/Read/ReadVariableOp!output/kernel/Read/ReadVariableOpoutput/bias/Read/ReadVariableOpVariable/Read/ReadVariableOpConst*
Tin
	2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *)
f$R"
 __inference__traced_save_1900109
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamelayer-1/kernellayer-1/biasoutput/kerneloutput/biasVariable*
Tin

2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *,
f'R%
#__inference__traced_restore_1900134ݱ
?

?
D__inference_model_2_layer_call_and_return_conditional_losses_1899528
input_3
classifier_graph_2_1899516
classifier_graph_2_1899518
classifier_graph_2_1899520
classifier_graph_2_1899522
classifier_graph_2_1899524
identity??*classifier_graph_2/StatefulPartitionedCall?
*classifier_graph_2/StatefulPartitionedCallStatefulPartitionedCallinput_3classifier_graph_2_1899516classifier_graph_2_1899518classifier_graph_2_1899520classifier_graph_2_1899522classifier_graph_2_1899524*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_classifier_graph_2_layer_call_and_return_conditional_losses_18994852,
*classifier_graph_2/StatefulPartitionedCall?
IdentityIdentity3classifier_graph_2/StatefulPartitionedCall:output:0+^classifier_graph_2/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????	:::::2X
*classifier_graph_2/StatefulPartitionedCall*classifier_graph_2/StatefulPartitionedCall:P L
'
_output_shapes
:?????????	
!
_user_specified_name	input_3
?)
?
O__inference_classifier_graph_2_layer_call_and_return_conditional_losses_1899755
xC
?sequential_2_project_2_matrix_transpose_readvariableop_resource7
3sequential_2_layer_1_matmul_readvariableop_resource8
4sequential_2_layer_1_biasadd_readvariableop_resource6
2sequential_2_output_matmul_readvariableop_resource7
3sequential_2_output_biasadd_readvariableop_resource
identity??+sequential_2/layer-1/BiasAdd/ReadVariableOp?*sequential_2/layer-1/MatMul/ReadVariableOp?*sequential_2/output/BiasAdd/ReadVariableOp?)sequential_2/output/MatMul/ReadVariableOp?.sequential_2/project_2/matmul_1/ReadVariableOp?6sequential_2/project_2/matrix_transpose/ReadVariableOp?
6sequential_2/project_2/matrix_transpose/ReadVariableOpReadVariableOp?sequential_2_project_2_matrix_transpose_readvariableop_resource*
_output_shapes

:	*
dtype028
6sequential_2/project_2/matrix_transpose/ReadVariableOp?
6sequential_2/project_2/matrix_transpose/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       28
6sequential_2/project_2/matrix_transpose/transpose/perm?
1sequential_2/project_2/matrix_transpose/transpose	Transpose>sequential_2/project_2/matrix_transpose/ReadVariableOp:value:0?sequential_2/project_2/matrix_transpose/transpose/perm:output:0*
T0*
_output_shapes

:	23
1sequential_2/project_2/matrix_transpose/transpose?
sequential_2/project_2/matmulMatMulx5sequential_2/project_2/matrix_transpose/transpose:y:0*
T0*'
_output_shapes
:?????????2
sequential_2/project_2/matmul?
.sequential_2/project_2/matmul_1/ReadVariableOpReadVariableOp?sequential_2_project_2_matrix_transpose_readvariableop_resource*
_output_shapes

:	*
dtype020
.sequential_2/project_2/matmul_1/ReadVariableOp?
sequential_2/project_2/matmul_1MatMul'sequential_2/project_2/matmul:product:06sequential_2/project_2/matmul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	2!
sequential_2/project_2/matmul_1?
sequential_2/project_2/subSubx)sequential_2/project_2/matmul_1:product:0*
T0*'
_output_shapes
:?????????	2
sequential_2/project_2/sub?
*sequential_2/layer-1/MatMul/ReadVariableOpReadVariableOp3sequential_2_layer_1_matmul_readvariableop_resource*
_output_shapes

:	2*
dtype02,
*sequential_2/layer-1/MatMul/ReadVariableOp?
sequential_2/layer-1/MatMulMatMulsequential_2/project_2/sub:z:02sequential_2/layer-1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
sequential_2/layer-1/MatMul?
+sequential_2/layer-1/BiasAdd/ReadVariableOpReadVariableOp4sequential_2_layer_1_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02-
+sequential_2/layer-1/BiasAdd/ReadVariableOp?
sequential_2/layer-1/BiasAddBiasAdd%sequential_2/layer-1/MatMul:product:03sequential_2/layer-1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
sequential_2/layer-1/BiasAdd?
sequential_2/layer-1/ReluRelu%sequential_2/layer-1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????22
sequential_2/layer-1/Relu?
)sequential_2/output/MatMul/ReadVariableOpReadVariableOp2sequential_2_output_matmul_readvariableop_resource*
_output_shapes

:2*
dtype02+
)sequential_2/output/MatMul/ReadVariableOp?
sequential_2/output/MatMulMatMul'sequential_2/layer-1/Relu:activations:01sequential_2/output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential_2/output/MatMul?
*sequential_2/output/BiasAdd/ReadVariableOpReadVariableOp3sequential_2_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02,
*sequential_2/output/BiasAdd/ReadVariableOp?
sequential_2/output/BiasAddBiasAdd$sequential_2/output/MatMul:product:02sequential_2/output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential_2/output/BiasAdd?
sequential_2/output/SoftmaxSoftmax$sequential_2/output/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
sequential_2/output/Softmax?
IdentityIdentity%sequential_2/output/Softmax:softmax:0,^sequential_2/layer-1/BiasAdd/ReadVariableOp+^sequential_2/layer-1/MatMul/ReadVariableOp+^sequential_2/output/BiasAdd/ReadVariableOp*^sequential_2/output/MatMul/ReadVariableOp/^sequential_2/project_2/matmul_1/ReadVariableOp7^sequential_2/project_2/matrix_transpose/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????	:::::2Z
+sequential_2/layer-1/BiasAdd/ReadVariableOp+sequential_2/layer-1/BiasAdd/ReadVariableOp2X
*sequential_2/layer-1/MatMul/ReadVariableOp*sequential_2/layer-1/MatMul/ReadVariableOp2X
*sequential_2/output/BiasAdd/ReadVariableOp*sequential_2/output/BiasAdd/ReadVariableOp2V
)sequential_2/output/MatMul/ReadVariableOp)sequential_2/output/MatMul/ReadVariableOp2`
.sequential_2/project_2/matmul_1/ReadVariableOp.sequential_2/project_2/matmul_1/ReadVariableOp2p
6sequential_2/project_2/matrix_transpose/ReadVariableOp6sequential_2/project_2/matrix_transpose/ReadVariableOp:J F
'
_output_shapes
:?????????	

_user_specified_namex
?
?
#__inference__traced_restore_1900134
file_prefix#
assignvariableop_layer_1_kernel#
assignvariableop_1_layer_1_bias$
 assignvariableop_2_output_kernel"
assignvariableop_3_output_bias
assignvariableop_4_variable

identity_6??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_2?AssignVariableOp_3?AssignVariableOp_4?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*,
_output_shapes
::::::*
dtypes

22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOpassignvariableop_layer_1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOpassignvariableop_1_layer_1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp assignvariableop_2_output_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOpassignvariableop_3_output_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOpassignvariableop_4_variableIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?

Identity_5Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_5?

Identity_6IdentityIdentity_5:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4*
T0*
_output_shapes
: 2

Identity_6"!

identity_6Identity_6:output:0*)
_input_shapes
: :::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_4:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?)
?
O__inference_classifier_graph_2_layer_call_and_return_conditional_losses_1899729
xC
?sequential_2_project_2_matrix_transpose_readvariableop_resource7
3sequential_2_layer_1_matmul_readvariableop_resource8
4sequential_2_layer_1_biasadd_readvariableop_resource6
2sequential_2_output_matmul_readvariableop_resource7
3sequential_2_output_biasadd_readvariableop_resource
identity??+sequential_2/layer-1/BiasAdd/ReadVariableOp?*sequential_2/layer-1/MatMul/ReadVariableOp?*sequential_2/output/BiasAdd/ReadVariableOp?)sequential_2/output/MatMul/ReadVariableOp?.sequential_2/project_2/matmul_1/ReadVariableOp?6sequential_2/project_2/matrix_transpose/ReadVariableOp?
6sequential_2/project_2/matrix_transpose/ReadVariableOpReadVariableOp?sequential_2_project_2_matrix_transpose_readvariableop_resource*
_output_shapes

:	*
dtype028
6sequential_2/project_2/matrix_transpose/ReadVariableOp?
6sequential_2/project_2/matrix_transpose/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       28
6sequential_2/project_2/matrix_transpose/transpose/perm?
1sequential_2/project_2/matrix_transpose/transpose	Transpose>sequential_2/project_2/matrix_transpose/ReadVariableOp:value:0?sequential_2/project_2/matrix_transpose/transpose/perm:output:0*
T0*
_output_shapes

:	23
1sequential_2/project_2/matrix_transpose/transpose?
sequential_2/project_2/matmulMatMulx5sequential_2/project_2/matrix_transpose/transpose:y:0*
T0*'
_output_shapes
:?????????2
sequential_2/project_2/matmul?
.sequential_2/project_2/matmul_1/ReadVariableOpReadVariableOp?sequential_2_project_2_matrix_transpose_readvariableop_resource*
_output_shapes

:	*
dtype020
.sequential_2/project_2/matmul_1/ReadVariableOp?
sequential_2/project_2/matmul_1MatMul'sequential_2/project_2/matmul:product:06sequential_2/project_2/matmul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	2!
sequential_2/project_2/matmul_1?
sequential_2/project_2/subSubx)sequential_2/project_2/matmul_1:product:0*
T0*'
_output_shapes
:?????????	2
sequential_2/project_2/sub?
*sequential_2/layer-1/MatMul/ReadVariableOpReadVariableOp3sequential_2_layer_1_matmul_readvariableop_resource*
_output_shapes

:	2*
dtype02,
*sequential_2/layer-1/MatMul/ReadVariableOp?
sequential_2/layer-1/MatMulMatMulsequential_2/project_2/sub:z:02sequential_2/layer-1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
sequential_2/layer-1/MatMul?
+sequential_2/layer-1/BiasAdd/ReadVariableOpReadVariableOp4sequential_2_layer_1_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02-
+sequential_2/layer-1/BiasAdd/ReadVariableOp?
sequential_2/layer-1/BiasAddBiasAdd%sequential_2/layer-1/MatMul:product:03sequential_2/layer-1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
sequential_2/layer-1/BiasAdd?
sequential_2/layer-1/ReluRelu%sequential_2/layer-1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????22
sequential_2/layer-1/Relu?
)sequential_2/output/MatMul/ReadVariableOpReadVariableOp2sequential_2_output_matmul_readvariableop_resource*
_output_shapes

:2*
dtype02+
)sequential_2/output/MatMul/ReadVariableOp?
sequential_2/output/MatMulMatMul'sequential_2/layer-1/Relu:activations:01sequential_2/output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential_2/output/MatMul?
*sequential_2/output/BiasAdd/ReadVariableOpReadVariableOp3sequential_2_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02,
*sequential_2/output/BiasAdd/ReadVariableOp?
sequential_2/output/BiasAddBiasAdd$sequential_2/output/MatMul:product:02sequential_2/output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential_2/output/BiasAdd?
sequential_2/output/SoftmaxSoftmax$sequential_2/output/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
sequential_2/output/Softmax?
IdentityIdentity%sequential_2/output/Softmax:softmax:0,^sequential_2/layer-1/BiasAdd/ReadVariableOp+^sequential_2/layer-1/MatMul/ReadVariableOp+^sequential_2/output/BiasAdd/ReadVariableOp*^sequential_2/output/MatMul/ReadVariableOp/^sequential_2/project_2/matmul_1/ReadVariableOp7^sequential_2/project_2/matrix_transpose/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????	:::::2Z
+sequential_2/layer-1/BiasAdd/ReadVariableOp+sequential_2/layer-1/BiasAdd/ReadVariableOp2X
*sequential_2/layer-1/MatMul/ReadVariableOp*sequential_2/layer-1/MatMul/ReadVariableOp2X
*sequential_2/output/BiasAdd/ReadVariableOp*sequential_2/output/BiasAdd/ReadVariableOp2V
)sequential_2/output/MatMul/ReadVariableOp)sequential_2/output/MatMul/ReadVariableOp2`
.sequential_2/project_2/matmul_1/ReadVariableOp.sequential_2/project_2/matmul_1/ReadVariableOp2p
6sequential_2/project_2/matrix_transpose/ReadVariableOp6sequential_2/project_2/matrix_transpose/ReadVariableOp:J F
'
_output_shapes
:?????????	

_user_specified_namex
?)
?
O__inference_classifier_graph_2_layer_call_and_return_conditional_losses_1899811
input_1C
?sequential_2_project_2_matrix_transpose_readvariableop_resource7
3sequential_2_layer_1_matmul_readvariableop_resource8
4sequential_2_layer_1_biasadd_readvariableop_resource6
2sequential_2_output_matmul_readvariableop_resource7
3sequential_2_output_biasadd_readvariableop_resource
identity??+sequential_2/layer-1/BiasAdd/ReadVariableOp?*sequential_2/layer-1/MatMul/ReadVariableOp?*sequential_2/output/BiasAdd/ReadVariableOp?)sequential_2/output/MatMul/ReadVariableOp?.sequential_2/project_2/matmul_1/ReadVariableOp?6sequential_2/project_2/matrix_transpose/ReadVariableOp?
6sequential_2/project_2/matrix_transpose/ReadVariableOpReadVariableOp?sequential_2_project_2_matrix_transpose_readvariableop_resource*
_output_shapes

:	*
dtype028
6sequential_2/project_2/matrix_transpose/ReadVariableOp?
6sequential_2/project_2/matrix_transpose/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       28
6sequential_2/project_2/matrix_transpose/transpose/perm?
1sequential_2/project_2/matrix_transpose/transpose	Transpose>sequential_2/project_2/matrix_transpose/ReadVariableOp:value:0?sequential_2/project_2/matrix_transpose/transpose/perm:output:0*
T0*
_output_shapes

:	23
1sequential_2/project_2/matrix_transpose/transpose?
sequential_2/project_2/matmulMatMulinput_15sequential_2/project_2/matrix_transpose/transpose:y:0*
T0*'
_output_shapes
:?????????2
sequential_2/project_2/matmul?
.sequential_2/project_2/matmul_1/ReadVariableOpReadVariableOp?sequential_2_project_2_matrix_transpose_readvariableop_resource*
_output_shapes

:	*
dtype020
.sequential_2/project_2/matmul_1/ReadVariableOp?
sequential_2/project_2/matmul_1MatMul'sequential_2/project_2/matmul:product:06sequential_2/project_2/matmul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	2!
sequential_2/project_2/matmul_1?
sequential_2/project_2/subSubinput_1)sequential_2/project_2/matmul_1:product:0*
T0*'
_output_shapes
:?????????	2
sequential_2/project_2/sub?
*sequential_2/layer-1/MatMul/ReadVariableOpReadVariableOp3sequential_2_layer_1_matmul_readvariableop_resource*
_output_shapes

:	2*
dtype02,
*sequential_2/layer-1/MatMul/ReadVariableOp?
sequential_2/layer-1/MatMulMatMulsequential_2/project_2/sub:z:02sequential_2/layer-1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
sequential_2/layer-1/MatMul?
+sequential_2/layer-1/BiasAdd/ReadVariableOpReadVariableOp4sequential_2_layer_1_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02-
+sequential_2/layer-1/BiasAdd/ReadVariableOp?
sequential_2/layer-1/BiasAddBiasAdd%sequential_2/layer-1/MatMul:product:03sequential_2/layer-1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
sequential_2/layer-1/BiasAdd?
sequential_2/layer-1/ReluRelu%sequential_2/layer-1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????22
sequential_2/layer-1/Relu?
)sequential_2/output/MatMul/ReadVariableOpReadVariableOp2sequential_2_output_matmul_readvariableop_resource*
_output_shapes

:2*
dtype02+
)sequential_2/output/MatMul/ReadVariableOp?
sequential_2/output/MatMulMatMul'sequential_2/layer-1/Relu:activations:01sequential_2/output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential_2/output/MatMul?
*sequential_2/output/BiasAdd/ReadVariableOpReadVariableOp3sequential_2_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02,
*sequential_2/output/BiasAdd/ReadVariableOp?
sequential_2/output/BiasAddBiasAdd$sequential_2/output/MatMul:product:02sequential_2/output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential_2/output/BiasAdd?
sequential_2/output/SoftmaxSoftmax$sequential_2/output/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
sequential_2/output/Softmax?
IdentityIdentity%sequential_2/output/Softmax:softmax:0,^sequential_2/layer-1/BiasAdd/ReadVariableOp+^sequential_2/layer-1/MatMul/ReadVariableOp+^sequential_2/output/BiasAdd/ReadVariableOp*^sequential_2/output/MatMul/ReadVariableOp/^sequential_2/project_2/matmul_1/ReadVariableOp7^sequential_2/project_2/matrix_transpose/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????	:::::2Z
+sequential_2/layer-1/BiasAdd/ReadVariableOp+sequential_2/layer-1/BiasAdd/ReadVariableOp2X
*sequential_2/layer-1/MatMul/ReadVariableOp*sequential_2/layer-1/MatMul/ReadVariableOp2X
*sequential_2/output/BiasAdd/ReadVariableOp*sequential_2/output/BiasAdd/ReadVariableOp2V
)sequential_2/output/MatMul/ReadVariableOp)sequential_2/output/MatMul/ReadVariableOp2`
.sequential_2/project_2/matmul_1/ReadVariableOp.sequential_2/project_2/matmul_1/ReadVariableOp2p
6sequential_2/project_2/matrix_transpose/ReadVariableOp6sequential_2/project_2/matrix_transpose/ReadVariableOp:P L
'
_output_shapes
:?????????	
!
_user_specified_name	input_1
?
?
.__inference_sequential_2_layer_call_fn_1899949

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_sequential_2_layer_call_and_return_conditional_losses_18993362
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????	:::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????	
 
_user_specified_nameinputs
?
?
I__inference_sequential_2_layer_call_and_return_conditional_losses_1899304

inputs
project_2_1899290
layer_1_1899293
layer_1_1899295
output_1899298
output_1899300
identity??layer-1/StatefulPartitionedCall?output/StatefulPartitionedCall?!project_2/StatefulPartitionedCall?
!project_2/StatefulPartitionedCallStatefulPartitionedCallinputsproject_2_1899290*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????	*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_project_2_layer_call_and_return_conditional_losses_18991972#
!project_2/StatefulPartitionedCall?
layer-1/StatefulPartitionedCallStatefulPartitionedCall*project_2/StatefulPartitionedCall:output:0layer_1_1899293layer_1_1899295*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_layer-1_layer_call_and_return_conditional_losses_18992232!
layer-1/StatefulPartitionedCall?
output/StatefulPartitionedCallStatefulPartitionedCall(layer-1/StatefulPartitionedCall:output:0output_1899298output_1899300*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_output_layer_call_and_return_conditional_losses_18992502 
output/StatefulPartitionedCall?
IdentityIdentity'output/StatefulPartitionedCall:output:0 ^layer-1/StatefulPartitionedCall^output/StatefulPartitionedCall"^project_2/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????	:::::2B
layer-1/StatefulPartitionedCalllayer-1/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall2F
!project_2/StatefulPartitionedCall!project_2/StatefulPartitionedCall:O K
'
_output_shapes
:?????????	
 
_user_specified_nameinputs
?
?
F__inference_project_2_layer_call_and_return_conditional_losses_1899197
x,
(matrix_transpose_readvariableop_resource
identity??matmul_1/ReadVariableOp?matrix_transpose/ReadVariableOp?
matrix_transpose/ReadVariableOpReadVariableOp(matrix_transpose_readvariableop_resource*
_output_shapes

:	*
dtype02!
matrix_transpose/ReadVariableOp?
matrix_transpose/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2!
matrix_transpose/transpose/perm?
matrix_transpose/transpose	Transpose'matrix_transpose/ReadVariableOp:value:0(matrix_transpose/transpose/perm:output:0*
T0*
_output_shapes

:	2
matrix_transpose/transposeo
matmulMatMulxmatrix_transpose/transpose:y:0*
T0*'
_output_shapes
:?????????2
matmul?
matmul_1/ReadVariableOpReadVariableOp(matrix_transpose_readvariableop_resource*
_output_shapes

:	*
dtype02
matmul_1/ReadVariableOp?
matmul_1MatMulmatmul:product:0matmul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	2

matmul_1Z
subSubxmatmul_1:product:0*
T0*'
_output_shapes
:?????????	2
sub?
IdentityIdentitysub:z:0^matmul_1/ReadVariableOp ^matrix_transpose/ReadVariableOp*
T0*'
_output_shapes
:?????????	2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????	:22
matmul_1/ReadVariableOpmatmul_1/ReadVariableOp2B
matrix_transpose/ReadVariableOpmatrix_transpose/ReadVariableOp:J F
'
_output_shapes
:?????????	

_user_specified_namex
?

?
D__inference_model_2_layer_call_and_return_conditional_losses_1899543
input_3
classifier_graph_2_1899531
classifier_graph_2_1899533
classifier_graph_2_1899535
classifier_graph_2_1899537
classifier_graph_2_1899539
identity??*classifier_graph_2/StatefulPartitionedCall?
*classifier_graph_2/StatefulPartitionedCallStatefulPartitionedCallinput_3classifier_graph_2_1899531classifier_graph_2_1899533classifier_graph_2_1899535classifier_graph_2_1899537classifier_graph_2_1899539*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_classifier_graph_2_layer_call_and_return_conditional_losses_18994282,
*classifier_graph_2/StatefulPartitionedCall?
IdentityIdentity3classifier_graph_2/StatefulPartitionedCall:output:0+^classifier_graph_2/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????	:::::2X
*classifier_graph_2/StatefulPartitionedCall*classifier_graph_2/StatefulPartitionedCall:P L
'
_output_shapes
:?????????	
!
_user_specified_name	input_3
?
?
4__inference_classifier_graph_2_layer_call_fn_1899785
x
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_classifier_graph_2_layer_call_and_return_conditional_losses_18994282
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????	:::::22
StatefulPartitionedCallStatefulPartitionedCall:J F
'
_output_shapes
:?????????	

_user_specified_namex
?
?
.__inference_sequential_2_layer_call_fn_1900031
project_2_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallproject_2_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_sequential_2_layer_call_and_return_conditional_losses_18993362
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????	:::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
'
_output_shapes
:?????????	
)
_user_specified_nameproject_2_input
?
?
.__inference_sequential_2_layer_call_fn_1900016
project_2_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallproject_2_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_sequential_2_layer_call_and_return_conditional_losses_18993042
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????	:::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
'
_output_shapes
:?????????	
)
_user_specified_nameproject_2_input
?!
?
I__inference_sequential_2_layer_call_and_return_conditional_losses_1899975
project_2_input6
2project_2_matrix_transpose_readvariableop_resource*
&layer_1_matmul_readvariableop_resource+
'layer_1_biasadd_readvariableop_resource)
%output_matmul_readvariableop_resource*
&output_biasadd_readvariableop_resource
identity??layer-1/BiasAdd/ReadVariableOp?layer-1/MatMul/ReadVariableOp?output/BiasAdd/ReadVariableOp?output/MatMul/ReadVariableOp?!project_2/matmul_1/ReadVariableOp?)project_2/matrix_transpose/ReadVariableOp?
)project_2/matrix_transpose/ReadVariableOpReadVariableOp2project_2_matrix_transpose_readvariableop_resource*
_output_shapes

:	*
dtype02+
)project_2/matrix_transpose/ReadVariableOp?
)project_2/matrix_transpose/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2+
)project_2/matrix_transpose/transpose/perm?
$project_2/matrix_transpose/transpose	Transpose1project_2/matrix_transpose/ReadVariableOp:value:02project_2/matrix_transpose/transpose/perm:output:0*
T0*
_output_shapes

:	2&
$project_2/matrix_transpose/transpose?
project_2/matmulMatMulproject_2_input(project_2/matrix_transpose/transpose:y:0*
T0*'
_output_shapes
:?????????2
project_2/matmul?
!project_2/matmul_1/ReadVariableOpReadVariableOp2project_2_matrix_transpose_readvariableop_resource*
_output_shapes

:	*
dtype02#
!project_2/matmul_1/ReadVariableOp?
project_2/matmul_1MatMulproject_2/matmul:product:0)project_2/matmul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	2
project_2/matmul_1?
project_2/subSubproject_2_inputproject_2/matmul_1:product:0*
T0*'
_output_shapes
:?????????	2
project_2/sub?
layer-1/MatMul/ReadVariableOpReadVariableOp&layer_1_matmul_readvariableop_resource*
_output_shapes

:	2*
dtype02
layer-1/MatMul/ReadVariableOp?
layer-1/MatMulMatMulproject_2/sub:z:0%layer-1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
layer-1/MatMul?
layer-1/BiasAdd/ReadVariableOpReadVariableOp'layer_1_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02 
layer-1/BiasAdd/ReadVariableOp?
layer-1/BiasAddBiasAddlayer-1/MatMul:product:0&layer-1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
layer-1/BiasAddp
layer-1/ReluRelulayer-1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????22
layer-1/Relu?
output/MatMul/ReadVariableOpReadVariableOp%output_matmul_readvariableop_resource*
_output_shapes

:2*
dtype02
output/MatMul/ReadVariableOp?
output/MatMulMatMullayer-1/Relu:activations:0$output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
output/MatMul?
output/BiasAdd/ReadVariableOpReadVariableOp&output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
output/BiasAdd/ReadVariableOp?
output/BiasAddBiasAddoutput/MatMul:product:0%output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
output/BiasAddv
output/SoftmaxSoftmaxoutput/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
output/Softmax?
IdentityIdentityoutput/Softmax:softmax:0^layer-1/BiasAdd/ReadVariableOp^layer-1/MatMul/ReadVariableOp^output/BiasAdd/ReadVariableOp^output/MatMul/ReadVariableOp"^project_2/matmul_1/ReadVariableOp*^project_2/matrix_transpose/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????	:::::2@
layer-1/BiasAdd/ReadVariableOplayer-1/BiasAdd/ReadVariableOp2>
layer-1/MatMul/ReadVariableOplayer-1/MatMul/ReadVariableOp2>
output/BiasAdd/ReadVariableOpoutput/BiasAdd/ReadVariableOp2<
output/MatMul/ReadVariableOpoutput/MatMul/ReadVariableOp2F
!project_2/matmul_1/ReadVariableOp!project_2/matmul_1/ReadVariableOp2V
)project_2/matrix_transpose/ReadVariableOp)project_2/matrix_transpose/ReadVariableOp:X T
'
_output_shapes
:?????????	
)
_user_specified_nameproject_2_input
?
}
(__inference_output_layer_call_fn_1900071

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_output_layer_call_and_return_conditional_losses_18992502
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????2::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs
?
?
 __inference__traced_save_1900109
file_prefix-
)savev2_layer_1_kernel_read_readvariableop+
'savev2_layer_1_bias_read_readvariableop,
(savev2_output_kernel_read_readvariableop*
&savev2_output_bias_read_readvariableop'
#savev2_variable_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0)savev2_layer_1_kernel_read_readvariableop'savev2_layer_1_bias_read_readvariableop(savev2_output_kernel_read_readvariableop&savev2_output_bias_read_readvariableop#savev2_variable_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes

22
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*A
_input_shapes0
.: :	2:2:2::	: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:	2: 

_output_shapes
:2:$ 

_output_shapes

:2: 

_output_shapes
::$ 

_output_shapes

:	:

_output_shapes
: 
?	
?
C__inference_output_layer_call_and_return_conditional_losses_1899250

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:2*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Softmax?
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????2::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs
?
?
I__inference_sequential_2_layer_call_and_return_conditional_losses_1899336

inputs
project_2_1899322
layer_1_1899325
layer_1_1899327
output_1899330
output_1899332
identity??layer-1/StatefulPartitionedCall?output/StatefulPartitionedCall?!project_2/StatefulPartitionedCall?
!project_2/StatefulPartitionedCallStatefulPartitionedCallinputsproject_2_1899322*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????	*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_project_2_layer_call_and_return_conditional_losses_18991972#
!project_2/StatefulPartitionedCall?
layer-1/StatefulPartitionedCallStatefulPartitionedCall*project_2/StatefulPartitionedCall:output:0layer_1_1899325layer_1_1899327*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_layer-1_layer_call_and_return_conditional_losses_18992232!
layer-1/StatefulPartitionedCall?
output/StatefulPartitionedCallStatefulPartitionedCall(layer-1/StatefulPartitionedCall:output:0output_1899330output_1899332*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_output_layer_call_and_return_conditional_losses_18992502 
output/StatefulPartitionedCall?
IdentityIdentity'output/StatefulPartitionedCall:output:0 ^layer-1/StatefulPartitionedCall^output/StatefulPartitionedCall"^project_2/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????	:::::2B
layer-1/StatefulPartitionedCalllayer-1/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall2F
!project_2/StatefulPartitionedCall!project_2/StatefulPartitionedCall:O K
'
_output_shapes
:?????????	
 
_user_specified_nameinputs
?	
?
C__inference_output_layer_call_and_return_conditional_losses_1900062

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:2*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Softmax?
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????2::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs
?
?
)__inference_model_2_layer_call_fn_1899574
input_3
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_3unknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_model_2_layer_call_and_return_conditional_losses_18995612
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????	:::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????	
!
_user_specified_name	input_3
?!
?
I__inference_sequential_2_layer_call_and_return_conditional_losses_1899919

inputs6
2project_2_matrix_transpose_readvariableop_resource*
&layer_1_matmul_readvariableop_resource+
'layer_1_biasadd_readvariableop_resource)
%output_matmul_readvariableop_resource*
&output_biasadd_readvariableop_resource
identity??layer-1/BiasAdd/ReadVariableOp?layer-1/MatMul/ReadVariableOp?output/BiasAdd/ReadVariableOp?output/MatMul/ReadVariableOp?!project_2/matmul_1/ReadVariableOp?)project_2/matrix_transpose/ReadVariableOp?
)project_2/matrix_transpose/ReadVariableOpReadVariableOp2project_2_matrix_transpose_readvariableop_resource*
_output_shapes

:	*
dtype02+
)project_2/matrix_transpose/ReadVariableOp?
)project_2/matrix_transpose/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2+
)project_2/matrix_transpose/transpose/perm?
$project_2/matrix_transpose/transpose	Transpose1project_2/matrix_transpose/ReadVariableOp:value:02project_2/matrix_transpose/transpose/perm:output:0*
T0*
_output_shapes

:	2&
$project_2/matrix_transpose/transpose?
project_2/matmulMatMulinputs(project_2/matrix_transpose/transpose:y:0*
T0*'
_output_shapes
:?????????2
project_2/matmul?
!project_2/matmul_1/ReadVariableOpReadVariableOp2project_2_matrix_transpose_readvariableop_resource*
_output_shapes

:	*
dtype02#
!project_2/matmul_1/ReadVariableOp?
project_2/matmul_1MatMulproject_2/matmul:product:0)project_2/matmul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	2
project_2/matmul_1}
project_2/subSubinputsproject_2/matmul_1:product:0*
T0*'
_output_shapes
:?????????	2
project_2/sub?
layer-1/MatMul/ReadVariableOpReadVariableOp&layer_1_matmul_readvariableop_resource*
_output_shapes

:	2*
dtype02
layer-1/MatMul/ReadVariableOp?
layer-1/MatMulMatMulproject_2/sub:z:0%layer-1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
layer-1/MatMul?
layer-1/BiasAdd/ReadVariableOpReadVariableOp'layer_1_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02 
layer-1/BiasAdd/ReadVariableOp?
layer-1/BiasAddBiasAddlayer-1/MatMul:product:0&layer-1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
layer-1/BiasAddp
layer-1/ReluRelulayer-1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????22
layer-1/Relu?
output/MatMul/ReadVariableOpReadVariableOp%output_matmul_readvariableop_resource*
_output_shapes

:2*
dtype02
output/MatMul/ReadVariableOp?
output/MatMulMatMullayer-1/Relu:activations:0$output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
output/MatMul?
output/BiasAdd/ReadVariableOpReadVariableOp&output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
output/BiasAdd/ReadVariableOp?
output/BiasAddBiasAddoutput/MatMul:product:0%output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
output/BiasAddv
output/SoftmaxSoftmaxoutput/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
output/Softmax?
IdentityIdentityoutput/Softmax:softmax:0^layer-1/BiasAdd/ReadVariableOp^layer-1/MatMul/ReadVariableOp^output/BiasAdd/ReadVariableOp^output/MatMul/ReadVariableOp"^project_2/matmul_1/ReadVariableOp*^project_2/matrix_transpose/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????	:::::2@
layer-1/BiasAdd/ReadVariableOplayer-1/BiasAdd/ReadVariableOp2>
layer-1/MatMul/ReadVariableOplayer-1/MatMul/ReadVariableOp2>
output/BiasAdd/ReadVariableOpoutput/BiasAdd/ReadVariableOp2<
output/MatMul/ReadVariableOpoutput/MatMul/ReadVariableOp2F
!project_2/matmul_1/ReadVariableOp!project_2/matmul_1/ReadVariableOp2V
)project_2/matrix_transpose/ReadVariableOp)project_2/matrix_transpose/ReadVariableOp:O K
'
_output_shapes
:?????????	
 
_user_specified_nameinputs
?
?
%__inference_signature_wrapper_1899621
input_3
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_3unknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8? *+
f&R$
"__inference__wrapped_model_18991842
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????	:::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????	
!
_user_specified_name	input_3
?
?
)__inference_model_2_layer_call_fn_1899604
input_3
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_3unknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_model_2_layer_call_and_return_conditional_losses_18995912
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????	:::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????	
!
_user_specified_name	input_3
?
?
4__inference_classifier_graph_2_layer_call_fn_1899867
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_classifier_graph_2_layer_call_and_return_conditional_losses_18994282
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????	:::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????	
!
_user_specified_name	input_1
?
?
)__inference_model_2_layer_call_fn_1899688

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_model_2_layer_call_and_return_conditional_losses_18995612
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????	:::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????	
 
_user_specified_nameinputs
?!
?
I__inference_sequential_2_layer_call_and_return_conditional_losses_1899893

inputs6
2project_2_matrix_transpose_readvariableop_resource*
&layer_1_matmul_readvariableop_resource+
'layer_1_biasadd_readvariableop_resource)
%output_matmul_readvariableop_resource*
&output_biasadd_readvariableop_resource
identity??layer-1/BiasAdd/ReadVariableOp?layer-1/MatMul/ReadVariableOp?output/BiasAdd/ReadVariableOp?output/MatMul/ReadVariableOp?!project_2/matmul_1/ReadVariableOp?)project_2/matrix_transpose/ReadVariableOp?
)project_2/matrix_transpose/ReadVariableOpReadVariableOp2project_2_matrix_transpose_readvariableop_resource*
_output_shapes

:	*
dtype02+
)project_2/matrix_transpose/ReadVariableOp?
)project_2/matrix_transpose/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2+
)project_2/matrix_transpose/transpose/perm?
$project_2/matrix_transpose/transpose	Transpose1project_2/matrix_transpose/ReadVariableOp:value:02project_2/matrix_transpose/transpose/perm:output:0*
T0*
_output_shapes

:	2&
$project_2/matrix_transpose/transpose?
project_2/matmulMatMulinputs(project_2/matrix_transpose/transpose:y:0*
T0*'
_output_shapes
:?????????2
project_2/matmul?
!project_2/matmul_1/ReadVariableOpReadVariableOp2project_2_matrix_transpose_readvariableop_resource*
_output_shapes

:	*
dtype02#
!project_2/matmul_1/ReadVariableOp?
project_2/matmul_1MatMulproject_2/matmul:product:0)project_2/matmul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	2
project_2/matmul_1}
project_2/subSubinputsproject_2/matmul_1:product:0*
T0*'
_output_shapes
:?????????	2
project_2/sub?
layer-1/MatMul/ReadVariableOpReadVariableOp&layer_1_matmul_readvariableop_resource*
_output_shapes

:	2*
dtype02
layer-1/MatMul/ReadVariableOp?
layer-1/MatMulMatMulproject_2/sub:z:0%layer-1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
layer-1/MatMul?
layer-1/BiasAdd/ReadVariableOpReadVariableOp'layer_1_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02 
layer-1/BiasAdd/ReadVariableOp?
layer-1/BiasAddBiasAddlayer-1/MatMul:product:0&layer-1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
layer-1/BiasAddp
layer-1/ReluRelulayer-1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????22
layer-1/Relu?
output/MatMul/ReadVariableOpReadVariableOp%output_matmul_readvariableop_resource*
_output_shapes

:2*
dtype02
output/MatMul/ReadVariableOp?
output/MatMulMatMullayer-1/Relu:activations:0$output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
output/MatMul?
output/BiasAdd/ReadVariableOpReadVariableOp&output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
output/BiasAdd/ReadVariableOp?
output/BiasAddBiasAddoutput/MatMul:product:0%output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
output/BiasAddv
output/SoftmaxSoftmaxoutput/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
output/Softmax?
IdentityIdentityoutput/Softmax:softmax:0^layer-1/BiasAdd/ReadVariableOp^layer-1/MatMul/ReadVariableOp^output/BiasAdd/ReadVariableOp^output/MatMul/ReadVariableOp"^project_2/matmul_1/ReadVariableOp*^project_2/matrix_transpose/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????	:::::2@
layer-1/BiasAdd/ReadVariableOplayer-1/BiasAdd/ReadVariableOp2>
layer-1/MatMul/ReadVariableOplayer-1/MatMul/ReadVariableOp2>
output/BiasAdd/ReadVariableOpoutput/BiasAdd/ReadVariableOp2<
output/MatMul/ReadVariableOpoutput/MatMul/ReadVariableOp2F
!project_2/matmul_1/ReadVariableOp!project_2/matmul_1/ReadVariableOp2V
)project_2/matrix_transpose/ReadVariableOp)project_2/matrix_transpose/ReadVariableOp:O K
'
_output_shapes
:?????????	
 
_user_specified_nameinputs
?6
?
D__inference_model_2_layer_call_and_return_conditional_losses_1899673

inputsV
Rclassifier_graph_2_sequential_2_project_2_matrix_transpose_readvariableop_resourceJ
Fclassifier_graph_2_sequential_2_layer_1_matmul_readvariableop_resourceK
Gclassifier_graph_2_sequential_2_layer_1_biasadd_readvariableop_resourceI
Eclassifier_graph_2_sequential_2_output_matmul_readvariableop_resourceJ
Fclassifier_graph_2_sequential_2_output_biasadd_readvariableop_resource
identity??>classifier_graph_2/sequential_2/layer-1/BiasAdd/ReadVariableOp?=classifier_graph_2/sequential_2/layer-1/MatMul/ReadVariableOp?=classifier_graph_2/sequential_2/output/BiasAdd/ReadVariableOp?<classifier_graph_2/sequential_2/output/MatMul/ReadVariableOp?Aclassifier_graph_2/sequential_2/project_2/matmul_1/ReadVariableOp?Iclassifier_graph_2/sequential_2/project_2/matrix_transpose/ReadVariableOp?
Iclassifier_graph_2/sequential_2/project_2/matrix_transpose/ReadVariableOpReadVariableOpRclassifier_graph_2_sequential_2_project_2_matrix_transpose_readvariableop_resource*
_output_shapes

:	*
dtype02K
Iclassifier_graph_2/sequential_2/project_2/matrix_transpose/ReadVariableOp?
Iclassifier_graph_2/sequential_2/project_2/matrix_transpose/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2K
Iclassifier_graph_2/sequential_2/project_2/matrix_transpose/transpose/perm?
Dclassifier_graph_2/sequential_2/project_2/matrix_transpose/transpose	TransposeQclassifier_graph_2/sequential_2/project_2/matrix_transpose/ReadVariableOp:value:0Rclassifier_graph_2/sequential_2/project_2/matrix_transpose/transpose/perm:output:0*
T0*
_output_shapes

:	2F
Dclassifier_graph_2/sequential_2/project_2/matrix_transpose/transpose?
0classifier_graph_2/sequential_2/project_2/matmulMatMulinputsHclassifier_graph_2/sequential_2/project_2/matrix_transpose/transpose:y:0*
T0*'
_output_shapes
:?????????22
0classifier_graph_2/sequential_2/project_2/matmul?
Aclassifier_graph_2/sequential_2/project_2/matmul_1/ReadVariableOpReadVariableOpRclassifier_graph_2_sequential_2_project_2_matrix_transpose_readvariableop_resource*
_output_shapes

:	*
dtype02C
Aclassifier_graph_2/sequential_2/project_2/matmul_1/ReadVariableOp?
2classifier_graph_2/sequential_2/project_2/matmul_1MatMul:classifier_graph_2/sequential_2/project_2/matmul:product:0Iclassifier_graph_2/sequential_2/project_2/matmul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	24
2classifier_graph_2/sequential_2/project_2/matmul_1?
-classifier_graph_2/sequential_2/project_2/subSubinputs<classifier_graph_2/sequential_2/project_2/matmul_1:product:0*
T0*'
_output_shapes
:?????????	2/
-classifier_graph_2/sequential_2/project_2/sub?
=classifier_graph_2/sequential_2/layer-1/MatMul/ReadVariableOpReadVariableOpFclassifier_graph_2_sequential_2_layer_1_matmul_readvariableop_resource*
_output_shapes

:	2*
dtype02?
=classifier_graph_2/sequential_2/layer-1/MatMul/ReadVariableOp?
.classifier_graph_2/sequential_2/layer-1/MatMulMatMul1classifier_graph_2/sequential_2/project_2/sub:z:0Eclassifier_graph_2/sequential_2/layer-1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????220
.classifier_graph_2/sequential_2/layer-1/MatMul?
>classifier_graph_2/sequential_2/layer-1/BiasAdd/ReadVariableOpReadVariableOpGclassifier_graph_2_sequential_2_layer_1_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02@
>classifier_graph_2/sequential_2/layer-1/BiasAdd/ReadVariableOp?
/classifier_graph_2/sequential_2/layer-1/BiasAddBiasAdd8classifier_graph_2/sequential_2/layer-1/MatMul:product:0Fclassifier_graph_2/sequential_2/layer-1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????221
/classifier_graph_2/sequential_2/layer-1/BiasAdd?
,classifier_graph_2/sequential_2/layer-1/ReluRelu8classifier_graph_2/sequential_2/layer-1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????22.
,classifier_graph_2/sequential_2/layer-1/Relu?
<classifier_graph_2/sequential_2/output/MatMul/ReadVariableOpReadVariableOpEclassifier_graph_2_sequential_2_output_matmul_readvariableop_resource*
_output_shapes

:2*
dtype02>
<classifier_graph_2/sequential_2/output/MatMul/ReadVariableOp?
-classifier_graph_2/sequential_2/output/MatMulMatMul:classifier_graph_2/sequential_2/layer-1/Relu:activations:0Dclassifier_graph_2/sequential_2/output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2/
-classifier_graph_2/sequential_2/output/MatMul?
=classifier_graph_2/sequential_2/output/BiasAdd/ReadVariableOpReadVariableOpFclassifier_graph_2_sequential_2_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02?
=classifier_graph_2/sequential_2/output/BiasAdd/ReadVariableOp?
.classifier_graph_2/sequential_2/output/BiasAddBiasAdd7classifier_graph_2/sequential_2/output/MatMul:product:0Eclassifier_graph_2/sequential_2/output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????20
.classifier_graph_2/sequential_2/output/BiasAdd?
.classifier_graph_2/sequential_2/output/SoftmaxSoftmax7classifier_graph_2/sequential_2/output/BiasAdd:output:0*
T0*'
_output_shapes
:?????????20
.classifier_graph_2/sequential_2/output/Softmax?
IdentityIdentity8classifier_graph_2/sequential_2/output/Softmax:softmax:0?^classifier_graph_2/sequential_2/layer-1/BiasAdd/ReadVariableOp>^classifier_graph_2/sequential_2/layer-1/MatMul/ReadVariableOp>^classifier_graph_2/sequential_2/output/BiasAdd/ReadVariableOp=^classifier_graph_2/sequential_2/output/MatMul/ReadVariableOpB^classifier_graph_2/sequential_2/project_2/matmul_1/ReadVariableOpJ^classifier_graph_2/sequential_2/project_2/matrix_transpose/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????	:::::2?
>classifier_graph_2/sequential_2/layer-1/BiasAdd/ReadVariableOp>classifier_graph_2/sequential_2/layer-1/BiasAdd/ReadVariableOp2~
=classifier_graph_2/sequential_2/layer-1/MatMul/ReadVariableOp=classifier_graph_2/sequential_2/layer-1/MatMul/ReadVariableOp2~
=classifier_graph_2/sequential_2/output/BiasAdd/ReadVariableOp=classifier_graph_2/sequential_2/output/BiasAdd/ReadVariableOp2|
<classifier_graph_2/sequential_2/output/MatMul/ReadVariableOp<classifier_graph_2/sequential_2/output/MatMul/ReadVariableOp2?
Aclassifier_graph_2/sequential_2/project_2/matmul_1/ReadVariableOpAclassifier_graph_2/sequential_2/project_2/matmul_1/ReadVariableOp2?
Iclassifier_graph_2/sequential_2/project_2/matrix_transpose/ReadVariableOpIclassifier_graph_2/sequential_2/project_2/matrix_transpose/ReadVariableOp:O K
'
_output_shapes
:?????????	
 
_user_specified_nameinputs
?
l
+__inference_project_2_layer_call_fn_1899205
x
unknown
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallxunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????	*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_project_2_layer_call_and_return_conditional_losses_18991972
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????	2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????	:22
StatefulPartitionedCallStatefulPartitionedCall:J F
'
_output_shapes
:?????????	

_user_specified_namex
?)
?
O__inference_classifier_graph_2_layer_call_and_return_conditional_losses_1899837
input_1C
?sequential_2_project_2_matrix_transpose_readvariableop_resource7
3sequential_2_layer_1_matmul_readvariableop_resource8
4sequential_2_layer_1_biasadd_readvariableop_resource6
2sequential_2_output_matmul_readvariableop_resource7
3sequential_2_output_biasadd_readvariableop_resource
identity??+sequential_2/layer-1/BiasAdd/ReadVariableOp?*sequential_2/layer-1/MatMul/ReadVariableOp?*sequential_2/output/BiasAdd/ReadVariableOp?)sequential_2/output/MatMul/ReadVariableOp?.sequential_2/project_2/matmul_1/ReadVariableOp?6sequential_2/project_2/matrix_transpose/ReadVariableOp?
6sequential_2/project_2/matrix_transpose/ReadVariableOpReadVariableOp?sequential_2_project_2_matrix_transpose_readvariableop_resource*
_output_shapes

:	*
dtype028
6sequential_2/project_2/matrix_transpose/ReadVariableOp?
6sequential_2/project_2/matrix_transpose/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       28
6sequential_2/project_2/matrix_transpose/transpose/perm?
1sequential_2/project_2/matrix_transpose/transpose	Transpose>sequential_2/project_2/matrix_transpose/ReadVariableOp:value:0?sequential_2/project_2/matrix_transpose/transpose/perm:output:0*
T0*
_output_shapes

:	23
1sequential_2/project_2/matrix_transpose/transpose?
sequential_2/project_2/matmulMatMulinput_15sequential_2/project_2/matrix_transpose/transpose:y:0*
T0*'
_output_shapes
:?????????2
sequential_2/project_2/matmul?
.sequential_2/project_2/matmul_1/ReadVariableOpReadVariableOp?sequential_2_project_2_matrix_transpose_readvariableop_resource*
_output_shapes

:	*
dtype020
.sequential_2/project_2/matmul_1/ReadVariableOp?
sequential_2/project_2/matmul_1MatMul'sequential_2/project_2/matmul:product:06sequential_2/project_2/matmul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	2!
sequential_2/project_2/matmul_1?
sequential_2/project_2/subSubinput_1)sequential_2/project_2/matmul_1:product:0*
T0*'
_output_shapes
:?????????	2
sequential_2/project_2/sub?
*sequential_2/layer-1/MatMul/ReadVariableOpReadVariableOp3sequential_2_layer_1_matmul_readvariableop_resource*
_output_shapes

:	2*
dtype02,
*sequential_2/layer-1/MatMul/ReadVariableOp?
sequential_2/layer-1/MatMulMatMulsequential_2/project_2/sub:z:02sequential_2/layer-1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
sequential_2/layer-1/MatMul?
+sequential_2/layer-1/BiasAdd/ReadVariableOpReadVariableOp4sequential_2_layer_1_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02-
+sequential_2/layer-1/BiasAdd/ReadVariableOp?
sequential_2/layer-1/BiasAddBiasAdd%sequential_2/layer-1/MatMul:product:03sequential_2/layer-1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
sequential_2/layer-1/BiasAdd?
sequential_2/layer-1/ReluRelu%sequential_2/layer-1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????22
sequential_2/layer-1/Relu?
)sequential_2/output/MatMul/ReadVariableOpReadVariableOp2sequential_2_output_matmul_readvariableop_resource*
_output_shapes

:2*
dtype02+
)sequential_2/output/MatMul/ReadVariableOp?
sequential_2/output/MatMulMatMul'sequential_2/layer-1/Relu:activations:01sequential_2/output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential_2/output/MatMul?
*sequential_2/output/BiasAdd/ReadVariableOpReadVariableOp3sequential_2_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02,
*sequential_2/output/BiasAdd/ReadVariableOp?
sequential_2/output/BiasAddBiasAdd$sequential_2/output/MatMul:product:02sequential_2/output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential_2/output/BiasAdd?
sequential_2/output/SoftmaxSoftmax$sequential_2/output/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
sequential_2/output/Softmax?
IdentityIdentity%sequential_2/output/Softmax:softmax:0,^sequential_2/layer-1/BiasAdd/ReadVariableOp+^sequential_2/layer-1/MatMul/ReadVariableOp+^sequential_2/output/BiasAdd/ReadVariableOp*^sequential_2/output/MatMul/ReadVariableOp/^sequential_2/project_2/matmul_1/ReadVariableOp7^sequential_2/project_2/matrix_transpose/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????	:::::2Z
+sequential_2/layer-1/BiasAdd/ReadVariableOp+sequential_2/layer-1/BiasAdd/ReadVariableOp2X
*sequential_2/layer-1/MatMul/ReadVariableOp*sequential_2/layer-1/MatMul/ReadVariableOp2X
*sequential_2/output/BiasAdd/ReadVariableOp*sequential_2/output/BiasAdd/ReadVariableOp2V
)sequential_2/output/MatMul/ReadVariableOp)sequential_2/output/MatMul/ReadVariableOp2`
.sequential_2/project_2/matmul_1/ReadVariableOp.sequential_2/project_2/matmul_1/ReadVariableOp2p
6sequential_2/project_2/matrix_transpose/ReadVariableOp6sequential_2/project_2/matrix_transpose/ReadVariableOp:P L
'
_output_shapes
:?????????	
!
_user_specified_name	input_1
?	
?
D__inference_layer-1_layer_call_and_return_conditional_losses_1899223

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:	2*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????22
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????	::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????	
 
_user_specified_nameinputs
?
~
)__inference_layer-1_layer_call_fn_1900051

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_layer-1_layer_call_and_return_conditional_losses_18992232
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????	::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????	
 
_user_specified_nameinputs
?;
?
"__inference__wrapped_model_1899184
input_3^
Zmodel_2_classifier_graph_2_sequential_2_project_2_matrix_transpose_readvariableop_resourceR
Nmodel_2_classifier_graph_2_sequential_2_layer_1_matmul_readvariableop_resourceS
Omodel_2_classifier_graph_2_sequential_2_layer_1_biasadd_readvariableop_resourceQ
Mmodel_2_classifier_graph_2_sequential_2_output_matmul_readvariableop_resourceR
Nmodel_2_classifier_graph_2_sequential_2_output_biasadd_readvariableop_resource
identity??Fmodel_2/classifier_graph_2/sequential_2/layer-1/BiasAdd/ReadVariableOp?Emodel_2/classifier_graph_2/sequential_2/layer-1/MatMul/ReadVariableOp?Emodel_2/classifier_graph_2/sequential_2/output/BiasAdd/ReadVariableOp?Dmodel_2/classifier_graph_2/sequential_2/output/MatMul/ReadVariableOp?Imodel_2/classifier_graph_2/sequential_2/project_2/matmul_1/ReadVariableOp?Qmodel_2/classifier_graph_2/sequential_2/project_2/matrix_transpose/ReadVariableOp?
Qmodel_2/classifier_graph_2/sequential_2/project_2/matrix_transpose/ReadVariableOpReadVariableOpZmodel_2_classifier_graph_2_sequential_2_project_2_matrix_transpose_readvariableop_resource*
_output_shapes

:	*
dtype02S
Qmodel_2/classifier_graph_2/sequential_2/project_2/matrix_transpose/ReadVariableOp?
Qmodel_2/classifier_graph_2/sequential_2/project_2/matrix_transpose/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2S
Qmodel_2/classifier_graph_2/sequential_2/project_2/matrix_transpose/transpose/perm?
Lmodel_2/classifier_graph_2/sequential_2/project_2/matrix_transpose/transpose	TransposeYmodel_2/classifier_graph_2/sequential_2/project_2/matrix_transpose/ReadVariableOp:value:0Zmodel_2/classifier_graph_2/sequential_2/project_2/matrix_transpose/transpose/perm:output:0*
T0*
_output_shapes

:	2N
Lmodel_2/classifier_graph_2/sequential_2/project_2/matrix_transpose/transpose?
8model_2/classifier_graph_2/sequential_2/project_2/matmulMatMulinput_3Pmodel_2/classifier_graph_2/sequential_2/project_2/matrix_transpose/transpose:y:0*
T0*'
_output_shapes
:?????????2:
8model_2/classifier_graph_2/sequential_2/project_2/matmul?
Imodel_2/classifier_graph_2/sequential_2/project_2/matmul_1/ReadVariableOpReadVariableOpZmodel_2_classifier_graph_2_sequential_2_project_2_matrix_transpose_readvariableop_resource*
_output_shapes

:	*
dtype02K
Imodel_2/classifier_graph_2/sequential_2/project_2/matmul_1/ReadVariableOp?
:model_2/classifier_graph_2/sequential_2/project_2/matmul_1MatMulBmodel_2/classifier_graph_2/sequential_2/project_2/matmul:product:0Qmodel_2/classifier_graph_2/sequential_2/project_2/matmul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	2<
:model_2/classifier_graph_2/sequential_2/project_2/matmul_1?
5model_2/classifier_graph_2/sequential_2/project_2/subSubinput_3Dmodel_2/classifier_graph_2/sequential_2/project_2/matmul_1:product:0*
T0*'
_output_shapes
:?????????	27
5model_2/classifier_graph_2/sequential_2/project_2/sub?
Emodel_2/classifier_graph_2/sequential_2/layer-1/MatMul/ReadVariableOpReadVariableOpNmodel_2_classifier_graph_2_sequential_2_layer_1_matmul_readvariableop_resource*
_output_shapes

:	2*
dtype02G
Emodel_2/classifier_graph_2/sequential_2/layer-1/MatMul/ReadVariableOp?
6model_2/classifier_graph_2/sequential_2/layer-1/MatMulMatMul9model_2/classifier_graph_2/sequential_2/project_2/sub:z:0Mmodel_2/classifier_graph_2/sequential_2/layer-1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????228
6model_2/classifier_graph_2/sequential_2/layer-1/MatMul?
Fmodel_2/classifier_graph_2/sequential_2/layer-1/BiasAdd/ReadVariableOpReadVariableOpOmodel_2_classifier_graph_2_sequential_2_layer_1_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02H
Fmodel_2/classifier_graph_2/sequential_2/layer-1/BiasAdd/ReadVariableOp?
7model_2/classifier_graph_2/sequential_2/layer-1/BiasAddBiasAdd@model_2/classifier_graph_2/sequential_2/layer-1/MatMul:product:0Nmodel_2/classifier_graph_2/sequential_2/layer-1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????229
7model_2/classifier_graph_2/sequential_2/layer-1/BiasAdd?
4model_2/classifier_graph_2/sequential_2/layer-1/ReluRelu@model_2/classifier_graph_2/sequential_2/layer-1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????226
4model_2/classifier_graph_2/sequential_2/layer-1/Relu?
Dmodel_2/classifier_graph_2/sequential_2/output/MatMul/ReadVariableOpReadVariableOpMmodel_2_classifier_graph_2_sequential_2_output_matmul_readvariableop_resource*
_output_shapes

:2*
dtype02F
Dmodel_2/classifier_graph_2/sequential_2/output/MatMul/ReadVariableOp?
5model_2/classifier_graph_2/sequential_2/output/MatMulMatMulBmodel_2/classifier_graph_2/sequential_2/layer-1/Relu:activations:0Lmodel_2/classifier_graph_2/sequential_2/output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????27
5model_2/classifier_graph_2/sequential_2/output/MatMul?
Emodel_2/classifier_graph_2/sequential_2/output/BiasAdd/ReadVariableOpReadVariableOpNmodel_2_classifier_graph_2_sequential_2_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02G
Emodel_2/classifier_graph_2/sequential_2/output/BiasAdd/ReadVariableOp?
6model_2/classifier_graph_2/sequential_2/output/BiasAddBiasAdd?model_2/classifier_graph_2/sequential_2/output/MatMul:product:0Mmodel_2/classifier_graph_2/sequential_2/output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????28
6model_2/classifier_graph_2/sequential_2/output/BiasAdd?
6model_2/classifier_graph_2/sequential_2/output/SoftmaxSoftmax?model_2/classifier_graph_2/sequential_2/output/BiasAdd:output:0*
T0*'
_output_shapes
:?????????28
6model_2/classifier_graph_2/sequential_2/output/Softmax?
IdentityIdentity@model_2/classifier_graph_2/sequential_2/output/Softmax:softmax:0G^model_2/classifier_graph_2/sequential_2/layer-1/BiasAdd/ReadVariableOpF^model_2/classifier_graph_2/sequential_2/layer-1/MatMul/ReadVariableOpF^model_2/classifier_graph_2/sequential_2/output/BiasAdd/ReadVariableOpE^model_2/classifier_graph_2/sequential_2/output/MatMul/ReadVariableOpJ^model_2/classifier_graph_2/sequential_2/project_2/matmul_1/ReadVariableOpR^model_2/classifier_graph_2/sequential_2/project_2/matrix_transpose/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????	:::::2?
Fmodel_2/classifier_graph_2/sequential_2/layer-1/BiasAdd/ReadVariableOpFmodel_2/classifier_graph_2/sequential_2/layer-1/BiasAdd/ReadVariableOp2?
Emodel_2/classifier_graph_2/sequential_2/layer-1/MatMul/ReadVariableOpEmodel_2/classifier_graph_2/sequential_2/layer-1/MatMul/ReadVariableOp2?
Emodel_2/classifier_graph_2/sequential_2/output/BiasAdd/ReadVariableOpEmodel_2/classifier_graph_2/sequential_2/output/BiasAdd/ReadVariableOp2?
Dmodel_2/classifier_graph_2/sequential_2/output/MatMul/ReadVariableOpDmodel_2/classifier_graph_2/sequential_2/output/MatMul/ReadVariableOp2?
Imodel_2/classifier_graph_2/sequential_2/project_2/matmul_1/ReadVariableOpImodel_2/classifier_graph_2/sequential_2/project_2/matmul_1/ReadVariableOp2?
Qmodel_2/classifier_graph_2/sequential_2/project_2/matrix_transpose/ReadVariableOpQmodel_2/classifier_graph_2/sequential_2/project_2/matrix_transpose/ReadVariableOp:P L
'
_output_shapes
:?????????	
!
_user_specified_name	input_3
?)
?
O__inference_classifier_graph_2_layer_call_and_return_conditional_losses_1899485
xC
?sequential_2_project_2_matrix_transpose_readvariableop_resource7
3sequential_2_layer_1_matmul_readvariableop_resource8
4sequential_2_layer_1_biasadd_readvariableop_resource6
2sequential_2_output_matmul_readvariableop_resource7
3sequential_2_output_biasadd_readvariableop_resource
identity??+sequential_2/layer-1/BiasAdd/ReadVariableOp?*sequential_2/layer-1/MatMul/ReadVariableOp?*sequential_2/output/BiasAdd/ReadVariableOp?)sequential_2/output/MatMul/ReadVariableOp?.sequential_2/project_2/matmul_1/ReadVariableOp?6sequential_2/project_2/matrix_transpose/ReadVariableOp?
6sequential_2/project_2/matrix_transpose/ReadVariableOpReadVariableOp?sequential_2_project_2_matrix_transpose_readvariableop_resource*
_output_shapes

:	*
dtype028
6sequential_2/project_2/matrix_transpose/ReadVariableOp?
6sequential_2/project_2/matrix_transpose/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       28
6sequential_2/project_2/matrix_transpose/transpose/perm?
1sequential_2/project_2/matrix_transpose/transpose	Transpose>sequential_2/project_2/matrix_transpose/ReadVariableOp:value:0?sequential_2/project_2/matrix_transpose/transpose/perm:output:0*
T0*
_output_shapes

:	23
1sequential_2/project_2/matrix_transpose/transpose?
sequential_2/project_2/matmulMatMulx5sequential_2/project_2/matrix_transpose/transpose:y:0*
T0*'
_output_shapes
:?????????2
sequential_2/project_2/matmul?
.sequential_2/project_2/matmul_1/ReadVariableOpReadVariableOp?sequential_2_project_2_matrix_transpose_readvariableop_resource*
_output_shapes

:	*
dtype020
.sequential_2/project_2/matmul_1/ReadVariableOp?
sequential_2/project_2/matmul_1MatMul'sequential_2/project_2/matmul:product:06sequential_2/project_2/matmul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	2!
sequential_2/project_2/matmul_1?
sequential_2/project_2/subSubx)sequential_2/project_2/matmul_1:product:0*
T0*'
_output_shapes
:?????????	2
sequential_2/project_2/sub?
*sequential_2/layer-1/MatMul/ReadVariableOpReadVariableOp3sequential_2_layer_1_matmul_readvariableop_resource*
_output_shapes

:	2*
dtype02,
*sequential_2/layer-1/MatMul/ReadVariableOp?
sequential_2/layer-1/MatMulMatMulsequential_2/project_2/sub:z:02sequential_2/layer-1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
sequential_2/layer-1/MatMul?
+sequential_2/layer-1/BiasAdd/ReadVariableOpReadVariableOp4sequential_2_layer_1_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02-
+sequential_2/layer-1/BiasAdd/ReadVariableOp?
sequential_2/layer-1/BiasAddBiasAdd%sequential_2/layer-1/MatMul:product:03sequential_2/layer-1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
sequential_2/layer-1/BiasAdd?
sequential_2/layer-1/ReluRelu%sequential_2/layer-1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????22
sequential_2/layer-1/Relu?
)sequential_2/output/MatMul/ReadVariableOpReadVariableOp2sequential_2_output_matmul_readvariableop_resource*
_output_shapes

:2*
dtype02+
)sequential_2/output/MatMul/ReadVariableOp?
sequential_2/output/MatMulMatMul'sequential_2/layer-1/Relu:activations:01sequential_2/output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential_2/output/MatMul?
*sequential_2/output/BiasAdd/ReadVariableOpReadVariableOp3sequential_2_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02,
*sequential_2/output/BiasAdd/ReadVariableOp?
sequential_2/output/BiasAddBiasAdd$sequential_2/output/MatMul:product:02sequential_2/output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential_2/output/BiasAdd?
sequential_2/output/SoftmaxSoftmax$sequential_2/output/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
sequential_2/output/Softmax?
IdentityIdentity%sequential_2/output/Softmax:softmax:0,^sequential_2/layer-1/BiasAdd/ReadVariableOp+^sequential_2/layer-1/MatMul/ReadVariableOp+^sequential_2/output/BiasAdd/ReadVariableOp*^sequential_2/output/MatMul/ReadVariableOp/^sequential_2/project_2/matmul_1/ReadVariableOp7^sequential_2/project_2/matrix_transpose/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????	:::::2Z
+sequential_2/layer-1/BiasAdd/ReadVariableOp+sequential_2/layer-1/BiasAdd/ReadVariableOp2X
*sequential_2/layer-1/MatMul/ReadVariableOp*sequential_2/layer-1/MatMul/ReadVariableOp2X
*sequential_2/output/BiasAdd/ReadVariableOp*sequential_2/output/BiasAdd/ReadVariableOp2V
)sequential_2/output/MatMul/ReadVariableOp)sequential_2/output/MatMul/ReadVariableOp2`
.sequential_2/project_2/matmul_1/ReadVariableOp.sequential_2/project_2/matmul_1/ReadVariableOp2p
6sequential_2/project_2/matrix_transpose/ReadVariableOp6sequential_2/project_2/matrix_transpose/ReadVariableOp:J F
'
_output_shapes
:?????????	

_user_specified_namex
?	
?
D__inference_layer-1_layer_call_and_return_conditional_losses_1900042

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:	2*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????22
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????	::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????	
 
_user_specified_nameinputs
?	
?
O__inference_classifier_graph_2_layer_call_and_return_conditional_losses_1899428
x
sequential_2_1899416
sequential_2_1899418
sequential_2_1899420
sequential_2_1899422
sequential_2_1899424
identity??$sequential_2/StatefulPartitionedCall?
$sequential_2/StatefulPartitionedCallStatefulPartitionedCallxsequential_2_1899416sequential_2_1899418sequential_2_1899420sequential_2_1899422sequential_2_1899424*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_sequential_2_layer_call_and_return_conditional_losses_18993362&
$sequential_2/StatefulPartitionedCall?
IdentityIdentity-sequential_2/StatefulPartitionedCall:output:0%^sequential_2/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????	:::::2L
$sequential_2/StatefulPartitionedCall$sequential_2/StatefulPartitionedCall:J F
'
_output_shapes
:?????????	

_user_specified_namex
?

?
D__inference_model_2_layer_call_and_return_conditional_losses_1899591

inputs
classifier_graph_2_1899579
classifier_graph_2_1899581
classifier_graph_2_1899583
classifier_graph_2_1899585
classifier_graph_2_1899587
identity??*classifier_graph_2/StatefulPartitionedCall?
*classifier_graph_2/StatefulPartitionedCallStatefulPartitionedCallinputsclassifier_graph_2_1899579classifier_graph_2_1899581classifier_graph_2_1899583classifier_graph_2_1899585classifier_graph_2_1899587*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_classifier_graph_2_layer_call_and_return_conditional_losses_18994282,
*classifier_graph_2/StatefulPartitionedCall?
IdentityIdentity3classifier_graph_2/StatefulPartitionedCall:output:0+^classifier_graph_2/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????	:::::2X
*classifier_graph_2/StatefulPartitionedCall*classifier_graph_2/StatefulPartitionedCall:O K
'
_output_shapes
:?????????	
 
_user_specified_nameinputs
?

?
D__inference_model_2_layer_call_and_return_conditional_losses_1899561

inputs
classifier_graph_2_1899549
classifier_graph_2_1899551
classifier_graph_2_1899553
classifier_graph_2_1899555
classifier_graph_2_1899557
identity??*classifier_graph_2/StatefulPartitionedCall?
*classifier_graph_2/StatefulPartitionedCallStatefulPartitionedCallinputsclassifier_graph_2_1899549classifier_graph_2_1899551classifier_graph_2_1899553classifier_graph_2_1899555classifier_graph_2_1899557*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_classifier_graph_2_layer_call_and_return_conditional_losses_18994852,
*classifier_graph_2/StatefulPartitionedCall?
IdentityIdentity3classifier_graph_2/StatefulPartitionedCall:output:0+^classifier_graph_2/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????	:::::2X
*classifier_graph_2/StatefulPartitionedCall*classifier_graph_2/StatefulPartitionedCall:O K
'
_output_shapes
:?????????	
 
_user_specified_nameinputs
?
?
4__inference_classifier_graph_2_layer_call_fn_1899770
x
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_classifier_graph_2_layer_call_and_return_conditional_losses_18994282
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????	:::::22
StatefulPartitionedCallStatefulPartitionedCall:J F
'
_output_shapes
:?????????	

_user_specified_namex
?6
?
D__inference_model_2_layer_call_and_return_conditional_losses_1899647

inputsV
Rclassifier_graph_2_sequential_2_project_2_matrix_transpose_readvariableop_resourceJ
Fclassifier_graph_2_sequential_2_layer_1_matmul_readvariableop_resourceK
Gclassifier_graph_2_sequential_2_layer_1_biasadd_readvariableop_resourceI
Eclassifier_graph_2_sequential_2_output_matmul_readvariableop_resourceJ
Fclassifier_graph_2_sequential_2_output_biasadd_readvariableop_resource
identity??>classifier_graph_2/sequential_2/layer-1/BiasAdd/ReadVariableOp?=classifier_graph_2/sequential_2/layer-1/MatMul/ReadVariableOp?=classifier_graph_2/sequential_2/output/BiasAdd/ReadVariableOp?<classifier_graph_2/sequential_2/output/MatMul/ReadVariableOp?Aclassifier_graph_2/sequential_2/project_2/matmul_1/ReadVariableOp?Iclassifier_graph_2/sequential_2/project_2/matrix_transpose/ReadVariableOp?
Iclassifier_graph_2/sequential_2/project_2/matrix_transpose/ReadVariableOpReadVariableOpRclassifier_graph_2_sequential_2_project_2_matrix_transpose_readvariableop_resource*
_output_shapes

:	*
dtype02K
Iclassifier_graph_2/sequential_2/project_2/matrix_transpose/ReadVariableOp?
Iclassifier_graph_2/sequential_2/project_2/matrix_transpose/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2K
Iclassifier_graph_2/sequential_2/project_2/matrix_transpose/transpose/perm?
Dclassifier_graph_2/sequential_2/project_2/matrix_transpose/transpose	TransposeQclassifier_graph_2/sequential_2/project_2/matrix_transpose/ReadVariableOp:value:0Rclassifier_graph_2/sequential_2/project_2/matrix_transpose/transpose/perm:output:0*
T0*
_output_shapes

:	2F
Dclassifier_graph_2/sequential_2/project_2/matrix_transpose/transpose?
0classifier_graph_2/sequential_2/project_2/matmulMatMulinputsHclassifier_graph_2/sequential_2/project_2/matrix_transpose/transpose:y:0*
T0*'
_output_shapes
:?????????22
0classifier_graph_2/sequential_2/project_2/matmul?
Aclassifier_graph_2/sequential_2/project_2/matmul_1/ReadVariableOpReadVariableOpRclassifier_graph_2_sequential_2_project_2_matrix_transpose_readvariableop_resource*
_output_shapes

:	*
dtype02C
Aclassifier_graph_2/sequential_2/project_2/matmul_1/ReadVariableOp?
2classifier_graph_2/sequential_2/project_2/matmul_1MatMul:classifier_graph_2/sequential_2/project_2/matmul:product:0Iclassifier_graph_2/sequential_2/project_2/matmul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	24
2classifier_graph_2/sequential_2/project_2/matmul_1?
-classifier_graph_2/sequential_2/project_2/subSubinputs<classifier_graph_2/sequential_2/project_2/matmul_1:product:0*
T0*'
_output_shapes
:?????????	2/
-classifier_graph_2/sequential_2/project_2/sub?
=classifier_graph_2/sequential_2/layer-1/MatMul/ReadVariableOpReadVariableOpFclassifier_graph_2_sequential_2_layer_1_matmul_readvariableop_resource*
_output_shapes

:	2*
dtype02?
=classifier_graph_2/sequential_2/layer-1/MatMul/ReadVariableOp?
.classifier_graph_2/sequential_2/layer-1/MatMulMatMul1classifier_graph_2/sequential_2/project_2/sub:z:0Eclassifier_graph_2/sequential_2/layer-1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????220
.classifier_graph_2/sequential_2/layer-1/MatMul?
>classifier_graph_2/sequential_2/layer-1/BiasAdd/ReadVariableOpReadVariableOpGclassifier_graph_2_sequential_2_layer_1_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02@
>classifier_graph_2/sequential_2/layer-1/BiasAdd/ReadVariableOp?
/classifier_graph_2/sequential_2/layer-1/BiasAddBiasAdd8classifier_graph_2/sequential_2/layer-1/MatMul:product:0Fclassifier_graph_2/sequential_2/layer-1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????221
/classifier_graph_2/sequential_2/layer-1/BiasAdd?
,classifier_graph_2/sequential_2/layer-1/ReluRelu8classifier_graph_2/sequential_2/layer-1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????22.
,classifier_graph_2/sequential_2/layer-1/Relu?
<classifier_graph_2/sequential_2/output/MatMul/ReadVariableOpReadVariableOpEclassifier_graph_2_sequential_2_output_matmul_readvariableop_resource*
_output_shapes

:2*
dtype02>
<classifier_graph_2/sequential_2/output/MatMul/ReadVariableOp?
-classifier_graph_2/sequential_2/output/MatMulMatMul:classifier_graph_2/sequential_2/layer-1/Relu:activations:0Dclassifier_graph_2/sequential_2/output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2/
-classifier_graph_2/sequential_2/output/MatMul?
=classifier_graph_2/sequential_2/output/BiasAdd/ReadVariableOpReadVariableOpFclassifier_graph_2_sequential_2_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02?
=classifier_graph_2/sequential_2/output/BiasAdd/ReadVariableOp?
.classifier_graph_2/sequential_2/output/BiasAddBiasAdd7classifier_graph_2/sequential_2/output/MatMul:product:0Eclassifier_graph_2/sequential_2/output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????20
.classifier_graph_2/sequential_2/output/BiasAdd?
.classifier_graph_2/sequential_2/output/SoftmaxSoftmax7classifier_graph_2/sequential_2/output/BiasAdd:output:0*
T0*'
_output_shapes
:?????????20
.classifier_graph_2/sequential_2/output/Softmax?
IdentityIdentity8classifier_graph_2/sequential_2/output/Softmax:softmax:0?^classifier_graph_2/sequential_2/layer-1/BiasAdd/ReadVariableOp>^classifier_graph_2/sequential_2/layer-1/MatMul/ReadVariableOp>^classifier_graph_2/sequential_2/output/BiasAdd/ReadVariableOp=^classifier_graph_2/sequential_2/output/MatMul/ReadVariableOpB^classifier_graph_2/sequential_2/project_2/matmul_1/ReadVariableOpJ^classifier_graph_2/sequential_2/project_2/matrix_transpose/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????	:::::2?
>classifier_graph_2/sequential_2/layer-1/BiasAdd/ReadVariableOp>classifier_graph_2/sequential_2/layer-1/BiasAdd/ReadVariableOp2~
=classifier_graph_2/sequential_2/layer-1/MatMul/ReadVariableOp=classifier_graph_2/sequential_2/layer-1/MatMul/ReadVariableOp2~
=classifier_graph_2/sequential_2/output/BiasAdd/ReadVariableOp=classifier_graph_2/sequential_2/output/BiasAdd/ReadVariableOp2|
<classifier_graph_2/sequential_2/output/MatMul/ReadVariableOp<classifier_graph_2/sequential_2/output/MatMul/ReadVariableOp2?
Aclassifier_graph_2/sequential_2/project_2/matmul_1/ReadVariableOpAclassifier_graph_2/sequential_2/project_2/matmul_1/ReadVariableOp2?
Iclassifier_graph_2/sequential_2/project_2/matrix_transpose/ReadVariableOpIclassifier_graph_2/sequential_2/project_2/matrix_transpose/ReadVariableOp:O K
'
_output_shapes
:?????????	
 
_user_specified_nameinputs
?!
?
I__inference_sequential_2_layer_call_and_return_conditional_losses_1900001
project_2_input6
2project_2_matrix_transpose_readvariableop_resource*
&layer_1_matmul_readvariableop_resource+
'layer_1_biasadd_readvariableop_resource)
%output_matmul_readvariableop_resource*
&output_biasadd_readvariableop_resource
identity??layer-1/BiasAdd/ReadVariableOp?layer-1/MatMul/ReadVariableOp?output/BiasAdd/ReadVariableOp?output/MatMul/ReadVariableOp?!project_2/matmul_1/ReadVariableOp?)project_2/matrix_transpose/ReadVariableOp?
)project_2/matrix_transpose/ReadVariableOpReadVariableOp2project_2_matrix_transpose_readvariableop_resource*
_output_shapes

:	*
dtype02+
)project_2/matrix_transpose/ReadVariableOp?
)project_2/matrix_transpose/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2+
)project_2/matrix_transpose/transpose/perm?
$project_2/matrix_transpose/transpose	Transpose1project_2/matrix_transpose/ReadVariableOp:value:02project_2/matrix_transpose/transpose/perm:output:0*
T0*
_output_shapes

:	2&
$project_2/matrix_transpose/transpose?
project_2/matmulMatMulproject_2_input(project_2/matrix_transpose/transpose:y:0*
T0*'
_output_shapes
:?????????2
project_2/matmul?
!project_2/matmul_1/ReadVariableOpReadVariableOp2project_2_matrix_transpose_readvariableop_resource*
_output_shapes

:	*
dtype02#
!project_2/matmul_1/ReadVariableOp?
project_2/matmul_1MatMulproject_2/matmul:product:0)project_2/matmul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	2
project_2/matmul_1?
project_2/subSubproject_2_inputproject_2/matmul_1:product:0*
T0*'
_output_shapes
:?????????	2
project_2/sub?
layer-1/MatMul/ReadVariableOpReadVariableOp&layer_1_matmul_readvariableop_resource*
_output_shapes

:	2*
dtype02
layer-1/MatMul/ReadVariableOp?
layer-1/MatMulMatMulproject_2/sub:z:0%layer-1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
layer-1/MatMul?
layer-1/BiasAdd/ReadVariableOpReadVariableOp'layer_1_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02 
layer-1/BiasAdd/ReadVariableOp?
layer-1/BiasAddBiasAddlayer-1/MatMul:product:0&layer-1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
layer-1/BiasAddp
layer-1/ReluRelulayer-1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????22
layer-1/Relu?
output/MatMul/ReadVariableOpReadVariableOp%output_matmul_readvariableop_resource*
_output_shapes

:2*
dtype02
output/MatMul/ReadVariableOp?
output/MatMulMatMullayer-1/Relu:activations:0$output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
output/MatMul?
output/BiasAdd/ReadVariableOpReadVariableOp&output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
output/BiasAdd/ReadVariableOp?
output/BiasAddBiasAddoutput/MatMul:product:0%output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
output/BiasAddv
output/SoftmaxSoftmaxoutput/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
output/Softmax?
IdentityIdentityoutput/Softmax:softmax:0^layer-1/BiasAdd/ReadVariableOp^layer-1/MatMul/ReadVariableOp^output/BiasAdd/ReadVariableOp^output/MatMul/ReadVariableOp"^project_2/matmul_1/ReadVariableOp*^project_2/matrix_transpose/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????	:::::2@
layer-1/BiasAdd/ReadVariableOplayer-1/BiasAdd/ReadVariableOp2>
layer-1/MatMul/ReadVariableOplayer-1/MatMul/ReadVariableOp2>
output/BiasAdd/ReadVariableOpoutput/BiasAdd/ReadVariableOp2<
output/MatMul/ReadVariableOpoutput/MatMul/ReadVariableOp2F
!project_2/matmul_1/ReadVariableOp!project_2/matmul_1/ReadVariableOp2V
)project_2/matrix_transpose/ReadVariableOp)project_2/matrix_transpose/ReadVariableOp:X T
'
_output_shapes
:?????????	
)
_user_specified_nameproject_2_input
?
?
4__inference_classifier_graph_2_layer_call_fn_1899852
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_classifier_graph_2_layer_call_and_return_conditional_losses_18994282
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????	:::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????	
!
_user_specified_name	input_1
?
?
)__inference_model_2_layer_call_fn_1899703

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_model_2_layer_call_and_return_conditional_losses_18995912
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????	:::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????	
 
_user_specified_nameinputs
?
?
.__inference_sequential_2_layer_call_fn_1899934

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_sequential_2_layer_call_and_return_conditional_losses_18993042
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????	:::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????	
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
;
input_30
serving_default_input_3:0?????????	F
classifier_graph_20
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
?

layer-0
layer_with_weights-0
layer-1
	variables
regularization_losses
trainable_variables
	keras_api

signatures
D_default_save_signature
*E&call_and_return_all_conditional_losses
F__call__"?
_tf_keras_network?{"class_name": "Functional", "name": "model_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "model_2", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 9]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_3"}, "name": "input_3", "inbound_nodes": []}, {"class_name": "ClassifierGraph", "config": {"layer was saved without config": true}, "name": "classifier_graph_2", "inbound_nodes": [[["input_3", 0, 0, {}]]]}], "input_layers": [["input_3", 0, 0]], "output_layers": [["classifier_graph_2", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 9]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 9]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional"}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input_3", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 9]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 9]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_3"}}
?

Layers
		model

	variables
regularization_losses
trainable_variables
	keras_api
*G&call_and_return_all_conditional_losses
H__call__"?
_tf_keras_model?{"class_name": "ClassifierGraph", "name": "classifier_graph_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "is_graph_network": false, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "ClassifierGraph"}}
C
0
1
2
3
4"
trackable_list_wrapper
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
?
layer_metrics
	variables
layer_regularization_losses
regularization_losses

layers
trainable_variables
non_trainable_variables
metrics
F__call__
D_default_save_signature
*E&call_and_return_all_conditional_losses
&E"call_and_return_conditional_losses"
_generic_user_object
,
Iserving_default"
signature_map
5
0
1
2"
trackable_list_wrapper
?
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
	variables
regularization_losses
trainable_variables
	keras_api
*J&call_and_return_all_conditional_losses
K__call__"?
_tf_keras_sequential?{"class_name": "Sequential", "name": "sequential_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_2", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 9]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "project_2_input"}}, {"class_name": "Project", "config": {"layer was saved without config": true}}, {"class_name": "Dense", "config": {"name": "layer-1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 9]}, "dtype": "float32", "units": 50, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "output", "trainable": true, "dtype": "float32", "units": 2, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 9]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 9]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential"}}
C
0
1
2
3
4"
trackable_list_wrapper
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
?
layer_metrics

	variables
 layer_regularization_losses
regularization_losses

!layers
trainable_variables
"non_trainable_variables
#metrics
H__call__
*G&call_and_return_all_conditional_losses
&G"call_and_return_conditional_losses"
_generic_user_object
 :	22layer-1/kernel
:22layer-1/bias
:22output/kernel
:2output/bias
:	2Variable
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
w
$	variables
%regularization_losses
&trainable_variables
'	keras_api
*L&call_and_return_all_conditional_losses
M__call__"?
_tf_keras_layer?{"class_name": "Project", "name": "project_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 9]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?

kernel
bias
(	variables
)regularization_losses
*trainable_variables
+	keras_api
*N&call_and_return_all_conditional_losses
O__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "layer-1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 9]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "layer-1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 9]}, "dtype": "float32", "units": 50, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 9}}}, "build_input_shape": {"class_name": "TensorShape", "items": [250, 9]}}
?

kernel
bias
,	variables
-regularization_losses
.trainable_variables
/	keras_api
*P&call_and_return_all_conditional_losses
Q__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "output", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "output", "trainable": true, "dtype": "float32", "units": 2, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 50}}}, "build_input_shape": {"class_name": "TensorShape", "items": [250, 50]}}
C
0
1
2
3
4"
trackable_list_wrapper
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
?
0layer_metrics
	variables
1layer_regularization_losses
regularization_losses

2layers
trainable_variables
3non_trainable_variables
4metrics
K__call__
*J&call_and_return_all_conditional_losses
&J"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
<
0
1
2
	3"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
5layer_metrics
$	variables
6layer_regularization_losses
%regularization_losses

7layers
&trainable_variables
8non_trainable_variables
9metrics
M__call__
*L&call_and_return_all_conditional_losses
&L"call_and_return_conditional_losses"
_generic_user_object
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
:layer_metrics
(	variables
;layer_regularization_losses
)regularization_losses

<layers
*trainable_variables
=non_trainable_variables
>metrics
O__call__
*N&call_and_return_all_conditional_losses
&N"call_and_return_conditional_losses"
_generic_user_object
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
?layer_metrics
,	variables
@layer_regularization_losses
-regularization_losses

Alayers
.trainable_variables
Bnon_trainable_variables
Cmetrics
Q__call__
*P&call_and_return_all_conditional_losses
&P"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?2?
"__inference__wrapped_model_1899184?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *&?#
!?
input_3?????????	
?2?
D__inference_model_2_layer_call_and_return_conditional_losses_1899647
D__inference_model_2_layer_call_and_return_conditional_losses_1899673
D__inference_model_2_layer_call_and_return_conditional_losses_1899528
D__inference_model_2_layer_call_and_return_conditional_losses_1899543?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
)__inference_model_2_layer_call_fn_1899703
)__inference_model_2_layer_call_fn_1899604
)__inference_model_2_layer_call_fn_1899688
)__inference_model_2_layer_call_fn_1899574?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
O__inference_classifier_graph_2_layer_call_and_return_conditional_losses_1899811
O__inference_classifier_graph_2_layer_call_and_return_conditional_losses_1899837
O__inference_classifier_graph_2_layer_call_and_return_conditional_losses_1899729
O__inference_classifier_graph_2_layer_call_and_return_conditional_losses_1899755?
???
FullArgSpec/
args'?$
jself
jx
	jpredict

jtraining
varargs
 
varkw
 
defaults?
p 
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
4__inference_classifier_graph_2_layer_call_fn_1899867
4__inference_classifier_graph_2_layer_call_fn_1899770
4__inference_classifier_graph_2_layer_call_fn_1899852
4__inference_classifier_graph_2_layer_call_fn_1899785?
???
FullArgSpec/
args'?$
jself
jx
	jpredict

jtraining
varargs
 
varkw
 
defaults?
p 
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
%__inference_signature_wrapper_1899621input_3"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
I__inference_sequential_2_layer_call_and_return_conditional_losses_1900001
I__inference_sequential_2_layer_call_and_return_conditional_losses_1899975
I__inference_sequential_2_layer_call_and_return_conditional_losses_1899919
I__inference_sequential_2_layer_call_and_return_conditional_losses_1899893?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
.__inference_sequential_2_layer_call_fn_1900016
.__inference_sequential_2_layer_call_fn_1900031
.__inference_sequential_2_layer_call_fn_1899934
.__inference_sequential_2_layer_call_fn_1899949?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
F__inference_project_2_layer_call_and_return_conditional_losses_1899197?
???
FullArgSpec
args?
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *?
??????????	
?2?
+__inference_project_2_layer_call_fn_1899205?
???
FullArgSpec
args?
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *?
??????????	
?2?
D__inference_layer-1_layer_call_and_return_conditional_losses_1900042?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_layer-1_layer_call_fn_1900051?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_output_layer_call_and_return_conditional_losses_1900062?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_output_layer_call_fn_1900071?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
"__inference__wrapped_model_1899184?0?-
&?#
!?
input_3?????????	
? "G?D
B
classifier_graph_2,?)
classifier_graph_2??????????
O__inference_classifier_graph_2_layer_call_and_return_conditional_losses_1899729b2?/
(?%
?
x?????????	
p 
p
? "%?"
?
0?????????
? ?
O__inference_classifier_graph_2_layer_call_and_return_conditional_losses_1899755b2?/
(?%
?
x?????????	
p 
p 
? "%?"
?
0?????????
? ?
O__inference_classifier_graph_2_layer_call_and_return_conditional_losses_1899811h8?5
.?+
!?
input_1?????????	
p 
p
? "%?"
?
0?????????
? ?
O__inference_classifier_graph_2_layer_call_and_return_conditional_losses_1899837h8?5
.?+
!?
input_1?????????	
p 
p 
? "%?"
?
0?????????
? ?
4__inference_classifier_graph_2_layer_call_fn_1899770U2?/
(?%
?
x?????????	
p 
p
? "???????????
4__inference_classifier_graph_2_layer_call_fn_1899785U2?/
(?%
?
x?????????	
p 
p 
? "???????????
4__inference_classifier_graph_2_layer_call_fn_1899852[8?5
.?+
!?
input_1?????????	
p 
p
? "???????????
4__inference_classifier_graph_2_layer_call_fn_1899867[8?5
.?+
!?
input_1?????????	
p 
p 
? "???????????
D__inference_layer-1_layer_call_and_return_conditional_losses_1900042\/?,
%?"
 ?
inputs?????????	
? "%?"
?
0?????????2
? |
)__inference_layer-1_layer_call_fn_1900051O/?,
%?"
 ?
inputs?????????	
? "??????????2?
D__inference_model_2_layer_call_and_return_conditional_losses_1899528h8?5
.?+
!?
input_3?????????	
p

 
? "%?"
?
0?????????
? ?
D__inference_model_2_layer_call_and_return_conditional_losses_1899543h8?5
.?+
!?
input_3?????????	
p 

 
? "%?"
?
0?????????
? ?
D__inference_model_2_layer_call_and_return_conditional_losses_1899647g7?4
-?*
 ?
inputs?????????	
p

 
? "%?"
?
0?????????
? ?
D__inference_model_2_layer_call_and_return_conditional_losses_1899673g7?4
-?*
 ?
inputs?????????	
p 

 
? "%?"
?
0?????????
? ?
)__inference_model_2_layer_call_fn_1899574[8?5
.?+
!?
input_3?????????	
p

 
? "???????????
)__inference_model_2_layer_call_fn_1899604[8?5
.?+
!?
input_3?????????	
p 

 
? "???????????
)__inference_model_2_layer_call_fn_1899688Z7?4
-?*
 ?
inputs?????????	
p

 
? "???????????
)__inference_model_2_layer_call_fn_1899703Z7?4
-?*
 ?
inputs?????????	
p 

 
? "???????????
C__inference_output_layer_call_and_return_conditional_losses_1900062\/?,
%?"
 ?
inputs?????????2
? "%?"
?
0?????????
? {
(__inference_output_layer_call_fn_1900071O/?,
%?"
 ?
inputs?????????2
? "???????????
F__inference_project_2_layer_call_and_return_conditional_losses_1899197V*?'
 ?
?
x?????????	
? "%?"
?
0?????????	
? x
+__inference_project_2_layer_call_fn_1899205I*?'
 ?
?
x?????????	
? "??????????	?
I__inference_sequential_2_layer_call_and_return_conditional_losses_1899893g7?4
-?*
 ?
inputs?????????	
p

 
? "%?"
?
0?????????
? ?
I__inference_sequential_2_layer_call_and_return_conditional_losses_1899919g7?4
-?*
 ?
inputs?????????	
p 

 
? "%?"
?
0?????????
? ?
I__inference_sequential_2_layer_call_and_return_conditional_losses_1899975p@?=
6?3
)?&
project_2_input?????????	
p

 
? "%?"
?
0?????????
? ?
I__inference_sequential_2_layer_call_and_return_conditional_losses_1900001p@?=
6?3
)?&
project_2_input?????????	
p 

 
? "%?"
?
0?????????
? ?
.__inference_sequential_2_layer_call_fn_1899934Z7?4
-?*
 ?
inputs?????????	
p

 
? "???????????
.__inference_sequential_2_layer_call_fn_1899949Z7?4
-?*
 ?
inputs?????????	
p 

 
? "???????????
.__inference_sequential_2_layer_call_fn_1900016c@?=
6?3
)?&
project_2_input?????????	
p

 
? "???????????
.__inference_sequential_2_layer_call_fn_1900031c@?=
6?3
)?&
project_2_input?????????	
p 

 
? "???????????
%__inference_signature_wrapper_1899621?;?8
? 
1?.
,
input_3!?
input_3?????????	"G?D
B
classifier_graph_2,?)
classifier_graph_2?????????