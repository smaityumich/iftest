??
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
serving_default_input_1Placeholder*'
_output_shapes
:?????????	*
dtype0*
shape:?????????	
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1Variablelayer-1/kernellayer-1/biasoutput/kerneloutput/bias*
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
GPU 2J 8? *-
f(R&
$__inference_signature_wrapper_632843
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
GPU 2J 8? *(
f#R!
__inference__traced_save_633331
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
GPU 2J 8? *+
f&R$
"__inference__traced_restore_633356??
?3
?
A__inference_model_layer_call_and_return_conditional_losses_632895

inputsP
Lclassifier_graph_sequential_project_matrix_transpose_readvariableop_resourceF
Bclassifier_graph_sequential_layer_1_matmul_readvariableop_resourceG
Cclassifier_graph_sequential_layer_1_biasadd_readvariableop_resourceE
Aclassifier_graph_sequential_output_matmul_readvariableop_resourceF
Bclassifier_graph_sequential_output_biasadd_readvariableop_resource
identity??:classifier_graph/sequential/layer-1/BiasAdd/ReadVariableOp?9classifier_graph/sequential/layer-1/MatMul/ReadVariableOp?9classifier_graph/sequential/output/BiasAdd/ReadVariableOp?8classifier_graph/sequential/output/MatMul/ReadVariableOp?;classifier_graph/sequential/project/matmul_1/ReadVariableOp?Cclassifier_graph/sequential/project/matrix_transpose/ReadVariableOp?
Cclassifier_graph/sequential/project/matrix_transpose/ReadVariableOpReadVariableOpLclassifier_graph_sequential_project_matrix_transpose_readvariableop_resource*
_output_shapes

:	*
dtype02E
Cclassifier_graph/sequential/project/matrix_transpose/ReadVariableOp?
Cclassifier_graph/sequential/project/matrix_transpose/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2E
Cclassifier_graph/sequential/project/matrix_transpose/transpose/perm?
>classifier_graph/sequential/project/matrix_transpose/transpose	TransposeKclassifier_graph/sequential/project/matrix_transpose/ReadVariableOp:value:0Lclassifier_graph/sequential/project/matrix_transpose/transpose/perm:output:0*
T0*
_output_shapes

:	2@
>classifier_graph/sequential/project/matrix_transpose/transpose?
*classifier_graph/sequential/project/matmulMatMulinputsBclassifier_graph/sequential/project/matrix_transpose/transpose:y:0*
T0*'
_output_shapes
:?????????2,
*classifier_graph/sequential/project/matmul?
;classifier_graph/sequential/project/matmul_1/ReadVariableOpReadVariableOpLclassifier_graph_sequential_project_matrix_transpose_readvariableop_resource*
_output_shapes

:	*
dtype02=
;classifier_graph/sequential/project/matmul_1/ReadVariableOp?
,classifier_graph/sequential/project/matmul_1MatMul4classifier_graph/sequential/project/matmul:product:0Cclassifier_graph/sequential/project/matmul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	2.
,classifier_graph/sequential/project/matmul_1?
'classifier_graph/sequential/project/subSubinputs6classifier_graph/sequential/project/matmul_1:product:0*
T0*'
_output_shapes
:?????????	2)
'classifier_graph/sequential/project/sub?
9classifier_graph/sequential/layer-1/MatMul/ReadVariableOpReadVariableOpBclassifier_graph_sequential_layer_1_matmul_readvariableop_resource*
_output_shapes

:	2*
dtype02;
9classifier_graph/sequential/layer-1/MatMul/ReadVariableOp?
*classifier_graph/sequential/layer-1/MatMulMatMul+classifier_graph/sequential/project/sub:z:0Aclassifier_graph/sequential/layer-1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22,
*classifier_graph/sequential/layer-1/MatMul?
:classifier_graph/sequential/layer-1/BiasAdd/ReadVariableOpReadVariableOpCclassifier_graph_sequential_layer_1_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02<
:classifier_graph/sequential/layer-1/BiasAdd/ReadVariableOp?
+classifier_graph/sequential/layer-1/BiasAddBiasAdd4classifier_graph/sequential/layer-1/MatMul:product:0Bclassifier_graph/sequential/layer-1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22-
+classifier_graph/sequential/layer-1/BiasAdd?
(classifier_graph/sequential/layer-1/ReluRelu4classifier_graph/sequential/layer-1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????22*
(classifier_graph/sequential/layer-1/Relu?
8classifier_graph/sequential/output/MatMul/ReadVariableOpReadVariableOpAclassifier_graph_sequential_output_matmul_readvariableop_resource*
_output_shapes

:2*
dtype02:
8classifier_graph/sequential/output/MatMul/ReadVariableOp?
)classifier_graph/sequential/output/MatMulMatMul6classifier_graph/sequential/layer-1/Relu:activations:0@classifier_graph/sequential/output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2+
)classifier_graph/sequential/output/MatMul?
9classifier_graph/sequential/output/BiasAdd/ReadVariableOpReadVariableOpBclassifier_graph_sequential_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02;
9classifier_graph/sequential/output/BiasAdd/ReadVariableOp?
*classifier_graph/sequential/output/BiasAddBiasAdd3classifier_graph/sequential/output/MatMul:product:0Aclassifier_graph/sequential/output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2,
*classifier_graph/sequential/output/BiasAdd?
*classifier_graph/sequential/output/SoftmaxSoftmax3classifier_graph/sequential/output/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2,
*classifier_graph/sequential/output/Softmax?
IdentityIdentity4classifier_graph/sequential/output/Softmax:softmax:0;^classifier_graph/sequential/layer-1/BiasAdd/ReadVariableOp:^classifier_graph/sequential/layer-1/MatMul/ReadVariableOp:^classifier_graph/sequential/output/BiasAdd/ReadVariableOp9^classifier_graph/sequential/output/MatMul/ReadVariableOp<^classifier_graph/sequential/project/matmul_1/ReadVariableOpD^classifier_graph/sequential/project/matrix_transpose/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????	:::::2x
:classifier_graph/sequential/layer-1/BiasAdd/ReadVariableOp:classifier_graph/sequential/layer-1/BiasAdd/ReadVariableOp2v
9classifier_graph/sequential/layer-1/MatMul/ReadVariableOp9classifier_graph/sequential/layer-1/MatMul/ReadVariableOp2v
9classifier_graph/sequential/output/BiasAdd/ReadVariableOp9classifier_graph/sequential/output/BiasAdd/ReadVariableOp2t
8classifier_graph/sequential/output/MatMul/ReadVariableOp8classifier_graph/sequential/output/MatMul/ReadVariableOp2z
;classifier_graph/sequential/project/matmul_1/ReadVariableOp;classifier_graph/sequential/project/matmul_1/ReadVariableOp2?
Cclassifier_graph/sequential/project/matrix_transpose/ReadVariableOpCclassifier_graph/sequential/project/matrix_transpose/ReadVariableOp:O K
'
_output_shapes
:?????????	
 
_user_specified_nameinputs
?
?
+__inference_sequential_layer_call_fn_633238

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
GPU 2J 8? *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_6325262
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
? 
?
F__inference_sequential_layer_call_and_return_conditional_losses_633141
project_input4
0project_matrix_transpose_readvariableop_resource*
&layer_1_matmul_readvariableop_resource+
'layer_1_biasadd_readvariableop_resource)
%output_matmul_readvariableop_resource*
&output_biasadd_readvariableop_resource
identity??layer-1/BiasAdd/ReadVariableOp?layer-1/MatMul/ReadVariableOp?output/BiasAdd/ReadVariableOp?output/MatMul/ReadVariableOp?project/matmul_1/ReadVariableOp?'project/matrix_transpose/ReadVariableOp?
'project/matrix_transpose/ReadVariableOpReadVariableOp0project_matrix_transpose_readvariableop_resource*
_output_shapes

:	*
dtype02)
'project/matrix_transpose/ReadVariableOp?
'project/matrix_transpose/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2)
'project/matrix_transpose/transpose/perm?
"project/matrix_transpose/transpose	Transpose/project/matrix_transpose/ReadVariableOp:value:00project/matrix_transpose/transpose/perm:output:0*
T0*
_output_shapes

:	2$
"project/matrix_transpose/transpose?
project/matmulMatMulproject_input&project/matrix_transpose/transpose:y:0*
T0*'
_output_shapes
:?????????2
project/matmul?
project/matmul_1/ReadVariableOpReadVariableOp0project_matrix_transpose_readvariableop_resource*
_output_shapes

:	*
dtype02!
project/matmul_1/ReadVariableOp?
project/matmul_1MatMulproject/matmul:product:0'project/matmul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	2
project/matmul_1~
project/subSubproject_inputproject/matmul_1:product:0*
T0*'
_output_shapes
:?????????	2
project/sub?
layer-1/MatMul/ReadVariableOpReadVariableOp&layer_1_matmul_readvariableop_resource*
_output_shapes

:	2*
dtype02
layer-1/MatMul/ReadVariableOp?
layer-1/MatMulMatMulproject/sub:z:0%layer-1/MatMul/ReadVariableOp:value:0*
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
IdentityIdentityoutput/Softmax:softmax:0^layer-1/BiasAdd/ReadVariableOp^layer-1/MatMul/ReadVariableOp^output/BiasAdd/ReadVariableOp^output/MatMul/ReadVariableOp ^project/matmul_1/ReadVariableOp(^project/matrix_transpose/ReadVariableOp*
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
output/MatMul/ReadVariableOpoutput/MatMul/ReadVariableOp2B
project/matmul_1/ReadVariableOpproject/matmul_1/ReadVariableOp2R
'project/matrix_transpose/ReadVariableOp'project/matrix_transpose/ReadVariableOp:V R
'
_output_shapes
:?????????	
'
_user_specified_nameproject_input
?
|
'__inference_output_layer_call_fn_633293

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
GPU 2J 8? *K
fFRD
B__inference_output_layer_call_and_return_conditional_losses_6324722
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
?
?
F__inference_sequential_layer_call_and_return_conditional_losses_632526

inputs
project_632512
layer_1_632515
layer_1_632517
output_632520
output_632522
identity??layer-1/StatefulPartitionedCall?output/StatefulPartitionedCall?project/StatefulPartitionedCall?
project/StatefulPartitionedCallStatefulPartitionedCallinputsproject_632512*
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
GPU 2J 8? *L
fGRE
C__inference_project_layer_call_and_return_conditional_losses_6324192!
project/StatefulPartitionedCall?
layer-1/StatefulPartitionedCallStatefulPartitionedCall(project/StatefulPartitionedCall:output:0layer_1_632515layer_1_632517*
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
GPU 2J 8? *L
fGRE
C__inference_layer-1_layer_call_and_return_conditional_losses_6324452!
layer-1/StatefulPartitionedCall?
output/StatefulPartitionedCallStatefulPartitionedCall(layer-1/StatefulPartitionedCall:output:0output_632520output_632522*
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
GPU 2J 8? *K
fFRD
B__inference_output_layer_call_and_return_conditional_losses_6324722 
output/StatefulPartitionedCall?
IdentityIdentity'output/StatefulPartitionedCall:output:0 ^layer-1/StatefulPartitionedCall^output/StatefulPartitionedCall ^project/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????	:::::2B
layer-1/StatefulPartitionedCalllayer-1/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall2B
project/StatefulPartitionedCallproject/StatefulPartitionedCall:O K
'
_output_shapes
:?????????	
 
_user_specified_nameinputs
?'
?
L__inference_classifier_graph_layer_call_and_return_conditional_losses_633033
x?
;sequential_project_matrix_transpose_readvariableop_resource5
1sequential_layer_1_matmul_readvariableop_resource6
2sequential_layer_1_biasadd_readvariableop_resource4
0sequential_output_matmul_readvariableop_resource5
1sequential_output_biasadd_readvariableop_resource
identity??)sequential/layer-1/BiasAdd/ReadVariableOp?(sequential/layer-1/MatMul/ReadVariableOp?(sequential/output/BiasAdd/ReadVariableOp?'sequential/output/MatMul/ReadVariableOp?*sequential/project/matmul_1/ReadVariableOp?2sequential/project/matrix_transpose/ReadVariableOp?
2sequential/project/matrix_transpose/ReadVariableOpReadVariableOp;sequential_project_matrix_transpose_readvariableop_resource*
_output_shapes

:	*
dtype024
2sequential/project/matrix_transpose/ReadVariableOp?
2sequential/project/matrix_transpose/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       24
2sequential/project/matrix_transpose/transpose/perm?
-sequential/project/matrix_transpose/transpose	Transpose:sequential/project/matrix_transpose/ReadVariableOp:value:0;sequential/project/matrix_transpose/transpose/perm:output:0*
T0*
_output_shapes

:	2/
-sequential/project/matrix_transpose/transpose?
sequential/project/matmulMatMulx1sequential/project/matrix_transpose/transpose:y:0*
T0*'
_output_shapes
:?????????2
sequential/project/matmul?
*sequential/project/matmul_1/ReadVariableOpReadVariableOp;sequential_project_matrix_transpose_readvariableop_resource*
_output_shapes

:	*
dtype02,
*sequential/project/matmul_1/ReadVariableOp?
sequential/project/matmul_1MatMul#sequential/project/matmul:product:02sequential/project/matmul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	2
sequential/project/matmul_1?
sequential/project/subSubx%sequential/project/matmul_1:product:0*
T0*'
_output_shapes
:?????????	2
sequential/project/sub?
(sequential/layer-1/MatMul/ReadVariableOpReadVariableOp1sequential_layer_1_matmul_readvariableop_resource*
_output_shapes

:	2*
dtype02*
(sequential/layer-1/MatMul/ReadVariableOp?
sequential/layer-1/MatMulMatMulsequential/project/sub:z:00sequential/layer-1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
sequential/layer-1/MatMul?
)sequential/layer-1/BiasAdd/ReadVariableOpReadVariableOp2sequential_layer_1_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02+
)sequential/layer-1/BiasAdd/ReadVariableOp?
sequential/layer-1/BiasAddBiasAdd#sequential/layer-1/MatMul:product:01sequential/layer-1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
sequential/layer-1/BiasAdd?
sequential/layer-1/ReluRelu#sequential/layer-1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????22
sequential/layer-1/Relu?
'sequential/output/MatMul/ReadVariableOpReadVariableOp0sequential_output_matmul_readvariableop_resource*
_output_shapes

:2*
dtype02)
'sequential/output/MatMul/ReadVariableOp?
sequential/output/MatMulMatMul%sequential/layer-1/Relu:activations:0/sequential/output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential/output/MatMul?
(sequential/output/BiasAdd/ReadVariableOpReadVariableOp1sequential_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(sequential/output/BiasAdd/ReadVariableOp?
sequential/output/BiasAddBiasAdd"sequential/output/MatMul:product:00sequential/output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential/output/BiasAdd?
sequential/output/SoftmaxSoftmax"sequential/output/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
sequential/output/Softmax?
IdentityIdentity#sequential/output/Softmax:softmax:0*^sequential/layer-1/BiasAdd/ReadVariableOp)^sequential/layer-1/MatMul/ReadVariableOp)^sequential/output/BiasAdd/ReadVariableOp(^sequential/output/MatMul/ReadVariableOp+^sequential/project/matmul_1/ReadVariableOp3^sequential/project/matrix_transpose/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????	:::::2V
)sequential/layer-1/BiasAdd/ReadVariableOp)sequential/layer-1/BiasAdd/ReadVariableOp2T
(sequential/layer-1/MatMul/ReadVariableOp(sequential/layer-1/MatMul/ReadVariableOp2T
(sequential/output/BiasAdd/ReadVariableOp(sequential/output/BiasAdd/ReadVariableOp2R
'sequential/output/MatMul/ReadVariableOp'sequential/output/MatMul/ReadVariableOp2X
*sequential/project/matmul_1/ReadVariableOp*sequential/project/matmul_1/ReadVariableOp2h
2sequential/project/matrix_transpose/ReadVariableOp2sequential/project/matrix_transpose/ReadVariableOp:J F
'
_output_shapes
:?????????	

_user_specified_namex
?
?
__inference__traced_save_633331
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
?
?
&__inference_model_layer_call_fn_632925

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
GPU 2J 8? *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_6328132
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
?	
?
C__inference_layer-1_layer_call_and_return_conditional_losses_633264

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
?7
?
!__inference__wrapped_model_632406
input_1V
Rmodel_classifier_graph_sequential_project_matrix_transpose_readvariableop_resourceL
Hmodel_classifier_graph_sequential_layer_1_matmul_readvariableop_resourceM
Imodel_classifier_graph_sequential_layer_1_biasadd_readvariableop_resourceK
Gmodel_classifier_graph_sequential_output_matmul_readvariableop_resourceL
Hmodel_classifier_graph_sequential_output_biasadd_readvariableop_resource
identity??@model/classifier_graph/sequential/layer-1/BiasAdd/ReadVariableOp??model/classifier_graph/sequential/layer-1/MatMul/ReadVariableOp??model/classifier_graph/sequential/output/BiasAdd/ReadVariableOp?>model/classifier_graph/sequential/output/MatMul/ReadVariableOp?Amodel/classifier_graph/sequential/project/matmul_1/ReadVariableOp?Imodel/classifier_graph/sequential/project/matrix_transpose/ReadVariableOp?
Imodel/classifier_graph/sequential/project/matrix_transpose/ReadVariableOpReadVariableOpRmodel_classifier_graph_sequential_project_matrix_transpose_readvariableop_resource*
_output_shapes

:	*
dtype02K
Imodel/classifier_graph/sequential/project/matrix_transpose/ReadVariableOp?
Imodel/classifier_graph/sequential/project/matrix_transpose/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2K
Imodel/classifier_graph/sequential/project/matrix_transpose/transpose/perm?
Dmodel/classifier_graph/sequential/project/matrix_transpose/transpose	TransposeQmodel/classifier_graph/sequential/project/matrix_transpose/ReadVariableOp:value:0Rmodel/classifier_graph/sequential/project/matrix_transpose/transpose/perm:output:0*
T0*
_output_shapes

:	2F
Dmodel/classifier_graph/sequential/project/matrix_transpose/transpose?
0model/classifier_graph/sequential/project/matmulMatMulinput_1Hmodel/classifier_graph/sequential/project/matrix_transpose/transpose:y:0*
T0*'
_output_shapes
:?????????22
0model/classifier_graph/sequential/project/matmul?
Amodel/classifier_graph/sequential/project/matmul_1/ReadVariableOpReadVariableOpRmodel_classifier_graph_sequential_project_matrix_transpose_readvariableop_resource*
_output_shapes

:	*
dtype02C
Amodel/classifier_graph/sequential/project/matmul_1/ReadVariableOp?
2model/classifier_graph/sequential/project/matmul_1MatMul:model/classifier_graph/sequential/project/matmul:product:0Imodel/classifier_graph/sequential/project/matmul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	24
2model/classifier_graph/sequential/project/matmul_1?
-model/classifier_graph/sequential/project/subSubinput_1<model/classifier_graph/sequential/project/matmul_1:product:0*
T0*'
_output_shapes
:?????????	2/
-model/classifier_graph/sequential/project/sub?
?model/classifier_graph/sequential/layer-1/MatMul/ReadVariableOpReadVariableOpHmodel_classifier_graph_sequential_layer_1_matmul_readvariableop_resource*
_output_shapes

:	2*
dtype02A
?model/classifier_graph/sequential/layer-1/MatMul/ReadVariableOp?
0model/classifier_graph/sequential/layer-1/MatMulMatMul1model/classifier_graph/sequential/project/sub:z:0Gmodel/classifier_graph/sequential/layer-1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????222
0model/classifier_graph/sequential/layer-1/MatMul?
@model/classifier_graph/sequential/layer-1/BiasAdd/ReadVariableOpReadVariableOpImodel_classifier_graph_sequential_layer_1_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02B
@model/classifier_graph/sequential/layer-1/BiasAdd/ReadVariableOp?
1model/classifier_graph/sequential/layer-1/BiasAddBiasAdd:model/classifier_graph/sequential/layer-1/MatMul:product:0Hmodel/classifier_graph/sequential/layer-1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????223
1model/classifier_graph/sequential/layer-1/BiasAdd?
.model/classifier_graph/sequential/layer-1/ReluRelu:model/classifier_graph/sequential/layer-1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????220
.model/classifier_graph/sequential/layer-1/Relu?
>model/classifier_graph/sequential/output/MatMul/ReadVariableOpReadVariableOpGmodel_classifier_graph_sequential_output_matmul_readvariableop_resource*
_output_shapes

:2*
dtype02@
>model/classifier_graph/sequential/output/MatMul/ReadVariableOp?
/model/classifier_graph/sequential/output/MatMulMatMul<model/classifier_graph/sequential/layer-1/Relu:activations:0Fmodel/classifier_graph/sequential/output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????21
/model/classifier_graph/sequential/output/MatMul?
?model/classifier_graph/sequential/output/BiasAdd/ReadVariableOpReadVariableOpHmodel_classifier_graph_sequential_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02A
?model/classifier_graph/sequential/output/BiasAdd/ReadVariableOp?
0model/classifier_graph/sequential/output/BiasAddBiasAdd9model/classifier_graph/sequential/output/MatMul:product:0Gmodel/classifier_graph/sequential/output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
0model/classifier_graph/sequential/output/BiasAdd?
0model/classifier_graph/sequential/output/SoftmaxSoftmax9model/classifier_graph/sequential/output/BiasAdd:output:0*
T0*'
_output_shapes
:?????????22
0model/classifier_graph/sequential/output/Softmax?
IdentityIdentity:model/classifier_graph/sequential/output/Softmax:softmax:0A^model/classifier_graph/sequential/layer-1/BiasAdd/ReadVariableOp@^model/classifier_graph/sequential/layer-1/MatMul/ReadVariableOp@^model/classifier_graph/sequential/output/BiasAdd/ReadVariableOp?^model/classifier_graph/sequential/output/MatMul/ReadVariableOpB^model/classifier_graph/sequential/project/matmul_1/ReadVariableOpJ^model/classifier_graph/sequential/project/matrix_transpose/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????	:::::2?
@model/classifier_graph/sequential/layer-1/BiasAdd/ReadVariableOp@model/classifier_graph/sequential/layer-1/BiasAdd/ReadVariableOp2?
?model/classifier_graph/sequential/layer-1/MatMul/ReadVariableOp?model/classifier_graph/sequential/layer-1/MatMul/ReadVariableOp2?
?model/classifier_graph/sequential/output/BiasAdd/ReadVariableOp?model/classifier_graph/sequential/output/BiasAdd/ReadVariableOp2?
>model/classifier_graph/sequential/output/MatMul/ReadVariableOp>model/classifier_graph/sequential/output/MatMul/ReadVariableOp2?
Amodel/classifier_graph/sequential/project/matmul_1/ReadVariableOpAmodel/classifier_graph/sequential/project/matmul_1/ReadVariableOp2?
Imodel/classifier_graph/sequential/project/matrix_transpose/ReadVariableOpImodel/classifier_graph/sequential/project/matrix_transpose/ReadVariableOp:P L
'
_output_shapes
:?????????	
!
_user_specified_name	input_1
?	
?
A__inference_model_layer_call_and_return_conditional_losses_632765
input_1
classifier_graph_632753
classifier_graph_632755
classifier_graph_632757
classifier_graph_632759
classifier_graph_632761
identity??(classifier_graph/StatefulPartitionedCall?
(classifier_graph/StatefulPartitionedCallStatefulPartitionedCallinput_1classifier_graph_632753classifier_graph_632755classifier_graph_632757classifier_graph_632759classifier_graph_632761*
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
GPU 2J 8? *U
fPRN
L__inference_classifier_graph_layer_call_and_return_conditional_losses_6326502*
(classifier_graph/StatefulPartitionedCall?
IdentityIdentity1classifier_graph/StatefulPartitionedCall:output:0)^classifier_graph/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????	:::::2T
(classifier_graph/StatefulPartitionedCall(classifier_graph/StatefulPartitionedCall:P L
'
_output_shapes
:?????????	
!
_user_specified_name	input_1
?	
?
C__inference_layer-1_layer_call_and_return_conditional_losses_632445

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
?
?
+__inference_sequential_layer_call_fn_633171
project_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallproject_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3*
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
GPU 2J 8? *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_6325582
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????	:::::22
StatefulPartitionedCallStatefulPartitionedCall:V R
'
_output_shapes
:?????????	
'
_user_specified_nameproject_input
?(
?
L__inference_classifier_graph_layer_call_and_return_conditional_losses_632951
input_1?
;sequential_project_matrix_transpose_readvariableop_resource5
1sequential_layer_1_matmul_readvariableop_resource6
2sequential_layer_1_biasadd_readvariableop_resource4
0sequential_output_matmul_readvariableop_resource5
1sequential_output_biasadd_readvariableop_resource
identity??)sequential/layer-1/BiasAdd/ReadVariableOp?(sequential/layer-1/MatMul/ReadVariableOp?(sequential/output/BiasAdd/ReadVariableOp?'sequential/output/MatMul/ReadVariableOp?*sequential/project/matmul_1/ReadVariableOp?2sequential/project/matrix_transpose/ReadVariableOp?
2sequential/project/matrix_transpose/ReadVariableOpReadVariableOp;sequential_project_matrix_transpose_readvariableop_resource*
_output_shapes

:	*
dtype024
2sequential/project/matrix_transpose/ReadVariableOp?
2sequential/project/matrix_transpose/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       24
2sequential/project/matrix_transpose/transpose/perm?
-sequential/project/matrix_transpose/transpose	Transpose:sequential/project/matrix_transpose/ReadVariableOp:value:0;sequential/project/matrix_transpose/transpose/perm:output:0*
T0*
_output_shapes

:	2/
-sequential/project/matrix_transpose/transpose?
sequential/project/matmulMatMulinput_11sequential/project/matrix_transpose/transpose:y:0*
T0*'
_output_shapes
:?????????2
sequential/project/matmul?
*sequential/project/matmul_1/ReadVariableOpReadVariableOp;sequential_project_matrix_transpose_readvariableop_resource*
_output_shapes

:	*
dtype02,
*sequential/project/matmul_1/ReadVariableOp?
sequential/project/matmul_1MatMul#sequential/project/matmul:product:02sequential/project/matmul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	2
sequential/project/matmul_1?
sequential/project/subSubinput_1%sequential/project/matmul_1:product:0*
T0*'
_output_shapes
:?????????	2
sequential/project/sub?
(sequential/layer-1/MatMul/ReadVariableOpReadVariableOp1sequential_layer_1_matmul_readvariableop_resource*
_output_shapes

:	2*
dtype02*
(sequential/layer-1/MatMul/ReadVariableOp?
sequential/layer-1/MatMulMatMulsequential/project/sub:z:00sequential/layer-1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
sequential/layer-1/MatMul?
)sequential/layer-1/BiasAdd/ReadVariableOpReadVariableOp2sequential_layer_1_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02+
)sequential/layer-1/BiasAdd/ReadVariableOp?
sequential/layer-1/BiasAddBiasAdd#sequential/layer-1/MatMul:product:01sequential/layer-1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
sequential/layer-1/BiasAdd?
sequential/layer-1/ReluRelu#sequential/layer-1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????22
sequential/layer-1/Relu?
'sequential/output/MatMul/ReadVariableOpReadVariableOp0sequential_output_matmul_readvariableop_resource*
_output_shapes

:2*
dtype02)
'sequential/output/MatMul/ReadVariableOp?
sequential/output/MatMulMatMul%sequential/layer-1/Relu:activations:0/sequential/output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential/output/MatMul?
(sequential/output/BiasAdd/ReadVariableOpReadVariableOp1sequential_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(sequential/output/BiasAdd/ReadVariableOp?
sequential/output/BiasAddBiasAdd"sequential/output/MatMul:product:00sequential/output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential/output/BiasAdd?
sequential/output/SoftmaxSoftmax"sequential/output/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
sequential/output/Softmax?
IdentityIdentity#sequential/output/Softmax:softmax:0*^sequential/layer-1/BiasAdd/ReadVariableOp)^sequential/layer-1/MatMul/ReadVariableOp)^sequential/output/BiasAdd/ReadVariableOp(^sequential/output/MatMul/ReadVariableOp+^sequential/project/matmul_1/ReadVariableOp3^sequential/project/matrix_transpose/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????	:::::2V
)sequential/layer-1/BiasAdd/ReadVariableOp)sequential/layer-1/BiasAdd/ReadVariableOp2T
(sequential/layer-1/MatMul/ReadVariableOp(sequential/layer-1/MatMul/ReadVariableOp2T
(sequential/output/BiasAdd/ReadVariableOp(sequential/output/BiasAdd/ReadVariableOp2R
'sequential/output/MatMul/ReadVariableOp'sequential/output/MatMul/ReadVariableOp2X
*sequential/project/matmul_1/ReadVariableOp*sequential/project/matmul_1/ReadVariableOp2h
2sequential/project/matrix_transpose/ReadVariableOp2sequential/project/matrix_transpose/ReadVariableOp:P L
'
_output_shapes
:?????????	
!
_user_specified_name	input_1
? 
?
F__inference_sequential_layer_call_and_return_conditional_losses_633223

inputs4
0project_matrix_transpose_readvariableop_resource*
&layer_1_matmul_readvariableop_resource+
'layer_1_biasadd_readvariableop_resource)
%output_matmul_readvariableop_resource*
&output_biasadd_readvariableop_resource
identity??layer-1/BiasAdd/ReadVariableOp?layer-1/MatMul/ReadVariableOp?output/BiasAdd/ReadVariableOp?output/MatMul/ReadVariableOp?project/matmul_1/ReadVariableOp?'project/matrix_transpose/ReadVariableOp?
'project/matrix_transpose/ReadVariableOpReadVariableOp0project_matrix_transpose_readvariableop_resource*
_output_shapes

:	*
dtype02)
'project/matrix_transpose/ReadVariableOp?
'project/matrix_transpose/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2)
'project/matrix_transpose/transpose/perm?
"project/matrix_transpose/transpose	Transpose/project/matrix_transpose/ReadVariableOp:value:00project/matrix_transpose/transpose/perm:output:0*
T0*
_output_shapes

:	2$
"project/matrix_transpose/transpose?
project/matmulMatMulinputs&project/matrix_transpose/transpose:y:0*
T0*'
_output_shapes
:?????????2
project/matmul?
project/matmul_1/ReadVariableOpReadVariableOp0project_matrix_transpose_readvariableop_resource*
_output_shapes

:	*
dtype02!
project/matmul_1/ReadVariableOp?
project/matmul_1MatMulproject/matmul:product:0'project/matmul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	2
project/matmul_1w
project/subSubinputsproject/matmul_1:product:0*
T0*'
_output_shapes
:?????????	2
project/sub?
layer-1/MatMul/ReadVariableOpReadVariableOp&layer_1_matmul_readvariableop_resource*
_output_shapes

:	2*
dtype02
layer-1/MatMul/ReadVariableOp?
layer-1/MatMulMatMulproject/sub:z:0%layer-1/MatMul/ReadVariableOp:value:0*
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
IdentityIdentityoutput/Softmax:softmax:0^layer-1/BiasAdd/ReadVariableOp^layer-1/MatMul/ReadVariableOp^output/BiasAdd/ReadVariableOp^output/MatMul/ReadVariableOp ^project/matmul_1/ReadVariableOp(^project/matrix_transpose/ReadVariableOp*
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
output/MatMul/ReadVariableOpoutput/MatMul/ReadVariableOp2B
project/matmul_1/ReadVariableOpproject/matmul_1/ReadVariableOp2R
'project/matrix_transpose/ReadVariableOp'project/matrix_transpose/ReadVariableOp:O K
'
_output_shapes
:?????????	
 
_user_specified_nameinputs
?
?
C__inference_project_layer_call_and_return_conditional_losses_632419
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
?
i
(__inference_project_layer_call_fn_632427
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
GPU 2J 8? *L
fGRE
C__inference_project_layer_call_and_return_conditional_losses_6324192
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
?
}
(__inference_layer-1_layer_call_fn_633273

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
GPU 2J 8? *L
fGRE
C__inference_layer-1_layer_call_and_return_conditional_losses_6324452
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
?(
?
L__inference_classifier_graph_layer_call_and_return_conditional_losses_632977
input_1?
;sequential_project_matrix_transpose_readvariableop_resource5
1sequential_layer_1_matmul_readvariableop_resource6
2sequential_layer_1_biasadd_readvariableop_resource4
0sequential_output_matmul_readvariableop_resource5
1sequential_output_biasadd_readvariableop_resource
identity??)sequential/layer-1/BiasAdd/ReadVariableOp?(sequential/layer-1/MatMul/ReadVariableOp?(sequential/output/BiasAdd/ReadVariableOp?'sequential/output/MatMul/ReadVariableOp?*sequential/project/matmul_1/ReadVariableOp?2sequential/project/matrix_transpose/ReadVariableOp?
2sequential/project/matrix_transpose/ReadVariableOpReadVariableOp;sequential_project_matrix_transpose_readvariableop_resource*
_output_shapes

:	*
dtype024
2sequential/project/matrix_transpose/ReadVariableOp?
2sequential/project/matrix_transpose/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       24
2sequential/project/matrix_transpose/transpose/perm?
-sequential/project/matrix_transpose/transpose	Transpose:sequential/project/matrix_transpose/ReadVariableOp:value:0;sequential/project/matrix_transpose/transpose/perm:output:0*
T0*
_output_shapes

:	2/
-sequential/project/matrix_transpose/transpose?
sequential/project/matmulMatMulinput_11sequential/project/matrix_transpose/transpose:y:0*
T0*'
_output_shapes
:?????????2
sequential/project/matmul?
*sequential/project/matmul_1/ReadVariableOpReadVariableOp;sequential_project_matrix_transpose_readvariableop_resource*
_output_shapes

:	*
dtype02,
*sequential/project/matmul_1/ReadVariableOp?
sequential/project/matmul_1MatMul#sequential/project/matmul:product:02sequential/project/matmul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	2
sequential/project/matmul_1?
sequential/project/subSubinput_1%sequential/project/matmul_1:product:0*
T0*'
_output_shapes
:?????????	2
sequential/project/sub?
(sequential/layer-1/MatMul/ReadVariableOpReadVariableOp1sequential_layer_1_matmul_readvariableop_resource*
_output_shapes

:	2*
dtype02*
(sequential/layer-1/MatMul/ReadVariableOp?
sequential/layer-1/MatMulMatMulsequential/project/sub:z:00sequential/layer-1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
sequential/layer-1/MatMul?
)sequential/layer-1/BiasAdd/ReadVariableOpReadVariableOp2sequential_layer_1_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02+
)sequential/layer-1/BiasAdd/ReadVariableOp?
sequential/layer-1/BiasAddBiasAdd#sequential/layer-1/MatMul:product:01sequential/layer-1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
sequential/layer-1/BiasAdd?
sequential/layer-1/ReluRelu#sequential/layer-1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????22
sequential/layer-1/Relu?
'sequential/output/MatMul/ReadVariableOpReadVariableOp0sequential_output_matmul_readvariableop_resource*
_output_shapes

:2*
dtype02)
'sequential/output/MatMul/ReadVariableOp?
sequential/output/MatMulMatMul%sequential/layer-1/Relu:activations:0/sequential/output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential/output/MatMul?
(sequential/output/BiasAdd/ReadVariableOpReadVariableOp1sequential_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(sequential/output/BiasAdd/ReadVariableOp?
sequential/output/BiasAddBiasAdd"sequential/output/MatMul:product:00sequential/output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential/output/BiasAdd?
sequential/output/SoftmaxSoftmax"sequential/output/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
sequential/output/Softmax?
IdentityIdentity#sequential/output/Softmax:softmax:0*^sequential/layer-1/BiasAdd/ReadVariableOp)^sequential/layer-1/MatMul/ReadVariableOp)^sequential/output/BiasAdd/ReadVariableOp(^sequential/output/MatMul/ReadVariableOp+^sequential/project/matmul_1/ReadVariableOp3^sequential/project/matrix_transpose/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????	:::::2V
)sequential/layer-1/BiasAdd/ReadVariableOp)sequential/layer-1/BiasAdd/ReadVariableOp2T
(sequential/layer-1/MatMul/ReadVariableOp(sequential/layer-1/MatMul/ReadVariableOp2T
(sequential/output/BiasAdd/ReadVariableOp(sequential/output/BiasAdd/ReadVariableOp2R
'sequential/output/MatMul/ReadVariableOp'sequential/output/MatMul/ReadVariableOp2X
*sequential/project/matmul_1/ReadVariableOp*sequential/project/matmul_1/ReadVariableOp2h
2sequential/project/matrix_transpose/ReadVariableOp2sequential/project/matrix_transpose/ReadVariableOp:P L
'
_output_shapes
:?????????	
!
_user_specified_name	input_1
?
?
1__inference_classifier_graph_layer_call_fn_633007
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
GPU 2J 8? *U
fPRN
L__inference_classifier_graph_layer_call_and_return_conditional_losses_6326502
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
?
?
L__inference_classifier_graph_layer_call_and_return_conditional_losses_632650
x
sequential_632638
sequential_632640
sequential_632642
sequential_632644
sequential_632646
identity??"sequential/StatefulPartitionedCall?
"sequential/StatefulPartitionedCallStatefulPartitionedCallxsequential_632638sequential_632640sequential_632642sequential_632644sequential_632646*
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
GPU 2J 8? *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_6325582$
"sequential/StatefulPartitionedCall?
IdentityIdentity+sequential/StatefulPartitionedCall:output:0#^sequential/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????	:::::2H
"sequential/StatefulPartitionedCall"sequential/StatefulPartitionedCall:J F
'
_output_shapes
:?????????	

_user_specified_namex
?
?
F__inference_sequential_layer_call_and_return_conditional_losses_632558

inputs
project_632544
layer_1_632547
layer_1_632549
output_632552
output_632554
identity??layer-1/StatefulPartitionedCall?output/StatefulPartitionedCall?project/StatefulPartitionedCall?
project/StatefulPartitionedCallStatefulPartitionedCallinputsproject_632544*
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
GPU 2J 8? *L
fGRE
C__inference_project_layer_call_and_return_conditional_losses_6324192!
project/StatefulPartitionedCall?
layer-1/StatefulPartitionedCallStatefulPartitionedCall(project/StatefulPartitionedCall:output:0layer_1_632547layer_1_632549*
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
GPU 2J 8? *L
fGRE
C__inference_layer-1_layer_call_and_return_conditional_losses_6324452!
layer-1/StatefulPartitionedCall?
output/StatefulPartitionedCallStatefulPartitionedCall(layer-1/StatefulPartitionedCall:output:0output_632552output_632554*
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
GPU 2J 8? *K
fFRD
B__inference_output_layer_call_and_return_conditional_losses_6324722 
output/StatefulPartitionedCall?
IdentityIdentity'output/StatefulPartitionedCall:output:0 ^layer-1/StatefulPartitionedCall^output/StatefulPartitionedCall ^project/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????	:::::2B
layer-1/StatefulPartitionedCalllayer-1/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall2B
project/StatefulPartitionedCallproject/StatefulPartitionedCall:O K
'
_output_shapes
:?????????	
 
_user_specified_nameinputs
?
?
1__inference_classifier_graph_layer_call_fn_632992
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
GPU 2J 8? *U
fPRN
L__inference_classifier_graph_layer_call_and_return_conditional_losses_6326502
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
?	
?
A__inference_model_layer_call_and_return_conditional_losses_632813

inputs
classifier_graph_632801
classifier_graph_632803
classifier_graph_632805
classifier_graph_632807
classifier_graph_632809
identity??(classifier_graph/StatefulPartitionedCall?
(classifier_graph/StatefulPartitionedCallStatefulPartitionedCallinputsclassifier_graph_632801classifier_graph_632803classifier_graph_632805classifier_graph_632807classifier_graph_632809*
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
GPU 2J 8? *U
fPRN
L__inference_classifier_graph_layer_call_and_return_conditional_losses_6326502*
(classifier_graph/StatefulPartitionedCall?
IdentityIdentity1classifier_graph/StatefulPartitionedCall:output:0)^classifier_graph/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????	:::::2T
(classifier_graph/StatefulPartitionedCall(classifier_graph/StatefulPartitionedCall:O K
'
_output_shapes
:?????????	
 
_user_specified_nameinputs
?
?
1__inference_classifier_graph_layer_call_fn_633089
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
GPU 2J 8? *U
fPRN
L__inference_classifier_graph_layer_call_and_return_conditional_losses_6326502
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
?3
?
A__inference_model_layer_call_and_return_conditional_losses_632869

inputsP
Lclassifier_graph_sequential_project_matrix_transpose_readvariableop_resourceF
Bclassifier_graph_sequential_layer_1_matmul_readvariableop_resourceG
Cclassifier_graph_sequential_layer_1_biasadd_readvariableop_resourceE
Aclassifier_graph_sequential_output_matmul_readvariableop_resourceF
Bclassifier_graph_sequential_output_biasadd_readvariableop_resource
identity??:classifier_graph/sequential/layer-1/BiasAdd/ReadVariableOp?9classifier_graph/sequential/layer-1/MatMul/ReadVariableOp?9classifier_graph/sequential/output/BiasAdd/ReadVariableOp?8classifier_graph/sequential/output/MatMul/ReadVariableOp?;classifier_graph/sequential/project/matmul_1/ReadVariableOp?Cclassifier_graph/sequential/project/matrix_transpose/ReadVariableOp?
Cclassifier_graph/sequential/project/matrix_transpose/ReadVariableOpReadVariableOpLclassifier_graph_sequential_project_matrix_transpose_readvariableop_resource*
_output_shapes

:	*
dtype02E
Cclassifier_graph/sequential/project/matrix_transpose/ReadVariableOp?
Cclassifier_graph/sequential/project/matrix_transpose/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2E
Cclassifier_graph/sequential/project/matrix_transpose/transpose/perm?
>classifier_graph/sequential/project/matrix_transpose/transpose	TransposeKclassifier_graph/sequential/project/matrix_transpose/ReadVariableOp:value:0Lclassifier_graph/sequential/project/matrix_transpose/transpose/perm:output:0*
T0*
_output_shapes

:	2@
>classifier_graph/sequential/project/matrix_transpose/transpose?
*classifier_graph/sequential/project/matmulMatMulinputsBclassifier_graph/sequential/project/matrix_transpose/transpose:y:0*
T0*'
_output_shapes
:?????????2,
*classifier_graph/sequential/project/matmul?
;classifier_graph/sequential/project/matmul_1/ReadVariableOpReadVariableOpLclassifier_graph_sequential_project_matrix_transpose_readvariableop_resource*
_output_shapes

:	*
dtype02=
;classifier_graph/sequential/project/matmul_1/ReadVariableOp?
,classifier_graph/sequential/project/matmul_1MatMul4classifier_graph/sequential/project/matmul:product:0Cclassifier_graph/sequential/project/matmul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	2.
,classifier_graph/sequential/project/matmul_1?
'classifier_graph/sequential/project/subSubinputs6classifier_graph/sequential/project/matmul_1:product:0*
T0*'
_output_shapes
:?????????	2)
'classifier_graph/sequential/project/sub?
9classifier_graph/sequential/layer-1/MatMul/ReadVariableOpReadVariableOpBclassifier_graph_sequential_layer_1_matmul_readvariableop_resource*
_output_shapes

:	2*
dtype02;
9classifier_graph/sequential/layer-1/MatMul/ReadVariableOp?
*classifier_graph/sequential/layer-1/MatMulMatMul+classifier_graph/sequential/project/sub:z:0Aclassifier_graph/sequential/layer-1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22,
*classifier_graph/sequential/layer-1/MatMul?
:classifier_graph/sequential/layer-1/BiasAdd/ReadVariableOpReadVariableOpCclassifier_graph_sequential_layer_1_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02<
:classifier_graph/sequential/layer-1/BiasAdd/ReadVariableOp?
+classifier_graph/sequential/layer-1/BiasAddBiasAdd4classifier_graph/sequential/layer-1/MatMul:product:0Bclassifier_graph/sequential/layer-1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22-
+classifier_graph/sequential/layer-1/BiasAdd?
(classifier_graph/sequential/layer-1/ReluRelu4classifier_graph/sequential/layer-1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????22*
(classifier_graph/sequential/layer-1/Relu?
8classifier_graph/sequential/output/MatMul/ReadVariableOpReadVariableOpAclassifier_graph_sequential_output_matmul_readvariableop_resource*
_output_shapes

:2*
dtype02:
8classifier_graph/sequential/output/MatMul/ReadVariableOp?
)classifier_graph/sequential/output/MatMulMatMul6classifier_graph/sequential/layer-1/Relu:activations:0@classifier_graph/sequential/output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2+
)classifier_graph/sequential/output/MatMul?
9classifier_graph/sequential/output/BiasAdd/ReadVariableOpReadVariableOpBclassifier_graph_sequential_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02;
9classifier_graph/sequential/output/BiasAdd/ReadVariableOp?
*classifier_graph/sequential/output/BiasAddBiasAdd3classifier_graph/sequential/output/MatMul:product:0Aclassifier_graph/sequential/output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2,
*classifier_graph/sequential/output/BiasAdd?
*classifier_graph/sequential/output/SoftmaxSoftmax3classifier_graph/sequential/output/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2,
*classifier_graph/sequential/output/Softmax?
IdentityIdentity4classifier_graph/sequential/output/Softmax:softmax:0;^classifier_graph/sequential/layer-1/BiasAdd/ReadVariableOp:^classifier_graph/sequential/layer-1/MatMul/ReadVariableOp:^classifier_graph/sequential/output/BiasAdd/ReadVariableOp9^classifier_graph/sequential/output/MatMul/ReadVariableOp<^classifier_graph/sequential/project/matmul_1/ReadVariableOpD^classifier_graph/sequential/project/matrix_transpose/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????	:::::2x
:classifier_graph/sequential/layer-1/BiasAdd/ReadVariableOp:classifier_graph/sequential/layer-1/BiasAdd/ReadVariableOp2v
9classifier_graph/sequential/layer-1/MatMul/ReadVariableOp9classifier_graph/sequential/layer-1/MatMul/ReadVariableOp2v
9classifier_graph/sequential/output/BiasAdd/ReadVariableOp9classifier_graph/sequential/output/BiasAdd/ReadVariableOp2t
8classifier_graph/sequential/output/MatMul/ReadVariableOp8classifier_graph/sequential/output/MatMul/ReadVariableOp2z
;classifier_graph/sequential/project/matmul_1/ReadVariableOp;classifier_graph/sequential/project/matmul_1/ReadVariableOp2?
Cclassifier_graph/sequential/project/matrix_transpose/ReadVariableOpCclassifier_graph/sequential/project/matrix_transpose/ReadVariableOp:O K
'
_output_shapes
:?????????	
 
_user_specified_nameinputs
?
?
$__inference_signature_wrapper_632843
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identity??StatefulPartitionedCall?
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
GPU 2J 8? **
f%R#
!__inference__wrapped_model_6324062
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
+__inference_sequential_layer_call_fn_633253

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
GPU 2J 8? *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_6325582
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
?	
?
A__inference_model_layer_call_and_return_conditional_losses_632750
input_1
classifier_graph_632738
classifier_graph_632740
classifier_graph_632742
classifier_graph_632744
classifier_graph_632746
identity??(classifier_graph/StatefulPartitionedCall?
(classifier_graph/StatefulPartitionedCallStatefulPartitionedCallinput_1classifier_graph_632738classifier_graph_632740classifier_graph_632742classifier_graph_632744classifier_graph_632746*
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
GPU 2J 8? *U
fPRN
L__inference_classifier_graph_layer_call_and_return_conditional_losses_6327072*
(classifier_graph/StatefulPartitionedCall?
IdentityIdentity1classifier_graph/StatefulPartitionedCall:output:0)^classifier_graph/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????	:::::2T
(classifier_graph/StatefulPartitionedCall(classifier_graph/StatefulPartitionedCall:P L
'
_output_shapes
:?????????	
!
_user_specified_name	input_1
?	
?
B__inference_output_layer_call_and_return_conditional_losses_632472

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
&__inference_model_layer_call_fn_632910

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
GPU 2J 8? *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_6327832
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
&__inference_model_layer_call_fn_632796
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
GPU 2J 8? *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_6327832
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
? 
?
F__inference_sequential_layer_call_and_return_conditional_losses_633197

inputs4
0project_matrix_transpose_readvariableop_resource*
&layer_1_matmul_readvariableop_resource+
'layer_1_biasadd_readvariableop_resource)
%output_matmul_readvariableop_resource*
&output_biasadd_readvariableop_resource
identity??layer-1/BiasAdd/ReadVariableOp?layer-1/MatMul/ReadVariableOp?output/BiasAdd/ReadVariableOp?output/MatMul/ReadVariableOp?project/matmul_1/ReadVariableOp?'project/matrix_transpose/ReadVariableOp?
'project/matrix_transpose/ReadVariableOpReadVariableOp0project_matrix_transpose_readvariableop_resource*
_output_shapes

:	*
dtype02)
'project/matrix_transpose/ReadVariableOp?
'project/matrix_transpose/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2)
'project/matrix_transpose/transpose/perm?
"project/matrix_transpose/transpose	Transpose/project/matrix_transpose/ReadVariableOp:value:00project/matrix_transpose/transpose/perm:output:0*
T0*
_output_shapes

:	2$
"project/matrix_transpose/transpose?
project/matmulMatMulinputs&project/matrix_transpose/transpose:y:0*
T0*'
_output_shapes
:?????????2
project/matmul?
project/matmul_1/ReadVariableOpReadVariableOp0project_matrix_transpose_readvariableop_resource*
_output_shapes

:	*
dtype02!
project/matmul_1/ReadVariableOp?
project/matmul_1MatMulproject/matmul:product:0'project/matmul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	2
project/matmul_1w
project/subSubinputsproject/matmul_1:product:0*
T0*'
_output_shapes
:?????????	2
project/sub?
layer-1/MatMul/ReadVariableOpReadVariableOp&layer_1_matmul_readvariableop_resource*
_output_shapes

:	2*
dtype02
layer-1/MatMul/ReadVariableOp?
layer-1/MatMulMatMulproject/sub:z:0%layer-1/MatMul/ReadVariableOp:value:0*
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
IdentityIdentityoutput/Softmax:softmax:0^layer-1/BiasAdd/ReadVariableOp^layer-1/MatMul/ReadVariableOp^output/BiasAdd/ReadVariableOp^output/MatMul/ReadVariableOp ^project/matmul_1/ReadVariableOp(^project/matrix_transpose/ReadVariableOp*
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
output/MatMul/ReadVariableOpoutput/MatMul/ReadVariableOp2B
project/matmul_1/ReadVariableOpproject/matmul_1/ReadVariableOp2R
'project/matrix_transpose/ReadVariableOp'project/matrix_transpose/ReadVariableOp:O K
'
_output_shapes
:?????????	
 
_user_specified_nameinputs
?
?
"__inference__traced_restore_633356
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
?'
?
L__inference_classifier_graph_layer_call_and_return_conditional_losses_633059
x?
;sequential_project_matrix_transpose_readvariableop_resource5
1sequential_layer_1_matmul_readvariableop_resource6
2sequential_layer_1_biasadd_readvariableop_resource4
0sequential_output_matmul_readvariableop_resource5
1sequential_output_biasadd_readvariableop_resource
identity??)sequential/layer-1/BiasAdd/ReadVariableOp?(sequential/layer-1/MatMul/ReadVariableOp?(sequential/output/BiasAdd/ReadVariableOp?'sequential/output/MatMul/ReadVariableOp?*sequential/project/matmul_1/ReadVariableOp?2sequential/project/matrix_transpose/ReadVariableOp?
2sequential/project/matrix_transpose/ReadVariableOpReadVariableOp;sequential_project_matrix_transpose_readvariableop_resource*
_output_shapes

:	*
dtype024
2sequential/project/matrix_transpose/ReadVariableOp?
2sequential/project/matrix_transpose/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       24
2sequential/project/matrix_transpose/transpose/perm?
-sequential/project/matrix_transpose/transpose	Transpose:sequential/project/matrix_transpose/ReadVariableOp:value:0;sequential/project/matrix_transpose/transpose/perm:output:0*
T0*
_output_shapes

:	2/
-sequential/project/matrix_transpose/transpose?
sequential/project/matmulMatMulx1sequential/project/matrix_transpose/transpose:y:0*
T0*'
_output_shapes
:?????????2
sequential/project/matmul?
*sequential/project/matmul_1/ReadVariableOpReadVariableOp;sequential_project_matrix_transpose_readvariableop_resource*
_output_shapes

:	*
dtype02,
*sequential/project/matmul_1/ReadVariableOp?
sequential/project/matmul_1MatMul#sequential/project/matmul:product:02sequential/project/matmul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	2
sequential/project/matmul_1?
sequential/project/subSubx%sequential/project/matmul_1:product:0*
T0*'
_output_shapes
:?????????	2
sequential/project/sub?
(sequential/layer-1/MatMul/ReadVariableOpReadVariableOp1sequential_layer_1_matmul_readvariableop_resource*
_output_shapes

:	2*
dtype02*
(sequential/layer-1/MatMul/ReadVariableOp?
sequential/layer-1/MatMulMatMulsequential/project/sub:z:00sequential/layer-1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
sequential/layer-1/MatMul?
)sequential/layer-1/BiasAdd/ReadVariableOpReadVariableOp2sequential_layer_1_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02+
)sequential/layer-1/BiasAdd/ReadVariableOp?
sequential/layer-1/BiasAddBiasAdd#sequential/layer-1/MatMul:product:01sequential/layer-1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
sequential/layer-1/BiasAdd?
sequential/layer-1/ReluRelu#sequential/layer-1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????22
sequential/layer-1/Relu?
'sequential/output/MatMul/ReadVariableOpReadVariableOp0sequential_output_matmul_readvariableop_resource*
_output_shapes

:2*
dtype02)
'sequential/output/MatMul/ReadVariableOp?
sequential/output/MatMulMatMul%sequential/layer-1/Relu:activations:0/sequential/output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential/output/MatMul?
(sequential/output/BiasAdd/ReadVariableOpReadVariableOp1sequential_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(sequential/output/BiasAdd/ReadVariableOp?
sequential/output/BiasAddBiasAdd"sequential/output/MatMul:product:00sequential/output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential/output/BiasAdd?
sequential/output/SoftmaxSoftmax"sequential/output/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
sequential/output/Softmax?
IdentityIdentity#sequential/output/Softmax:softmax:0*^sequential/layer-1/BiasAdd/ReadVariableOp)^sequential/layer-1/MatMul/ReadVariableOp)^sequential/output/BiasAdd/ReadVariableOp(^sequential/output/MatMul/ReadVariableOp+^sequential/project/matmul_1/ReadVariableOp3^sequential/project/matrix_transpose/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????	:::::2V
)sequential/layer-1/BiasAdd/ReadVariableOp)sequential/layer-1/BiasAdd/ReadVariableOp2T
(sequential/layer-1/MatMul/ReadVariableOp(sequential/layer-1/MatMul/ReadVariableOp2T
(sequential/output/BiasAdd/ReadVariableOp(sequential/output/BiasAdd/ReadVariableOp2R
'sequential/output/MatMul/ReadVariableOp'sequential/output/MatMul/ReadVariableOp2X
*sequential/project/matmul_1/ReadVariableOp*sequential/project/matmul_1/ReadVariableOp2h
2sequential/project/matrix_transpose/ReadVariableOp2sequential/project/matrix_transpose/ReadVariableOp:J F
'
_output_shapes
:?????????	

_user_specified_namex
?	
?
B__inference_output_layer_call_and_return_conditional_losses_633284

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
&__inference_model_layer_call_fn_632826
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
GPU 2J 8? *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_6328132
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
? 
?
F__inference_sequential_layer_call_and_return_conditional_losses_633115
project_input4
0project_matrix_transpose_readvariableop_resource*
&layer_1_matmul_readvariableop_resource+
'layer_1_biasadd_readvariableop_resource)
%output_matmul_readvariableop_resource*
&output_biasadd_readvariableop_resource
identity??layer-1/BiasAdd/ReadVariableOp?layer-1/MatMul/ReadVariableOp?output/BiasAdd/ReadVariableOp?output/MatMul/ReadVariableOp?project/matmul_1/ReadVariableOp?'project/matrix_transpose/ReadVariableOp?
'project/matrix_transpose/ReadVariableOpReadVariableOp0project_matrix_transpose_readvariableop_resource*
_output_shapes

:	*
dtype02)
'project/matrix_transpose/ReadVariableOp?
'project/matrix_transpose/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2)
'project/matrix_transpose/transpose/perm?
"project/matrix_transpose/transpose	Transpose/project/matrix_transpose/ReadVariableOp:value:00project/matrix_transpose/transpose/perm:output:0*
T0*
_output_shapes

:	2$
"project/matrix_transpose/transpose?
project/matmulMatMulproject_input&project/matrix_transpose/transpose:y:0*
T0*'
_output_shapes
:?????????2
project/matmul?
project/matmul_1/ReadVariableOpReadVariableOp0project_matrix_transpose_readvariableop_resource*
_output_shapes

:	*
dtype02!
project/matmul_1/ReadVariableOp?
project/matmul_1MatMulproject/matmul:product:0'project/matmul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	2
project/matmul_1~
project/subSubproject_inputproject/matmul_1:product:0*
T0*'
_output_shapes
:?????????	2
project/sub?
layer-1/MatMul/ReadVariableOpReadVariableOp&layer_1_matmul_readvariableop_resource*
_output_shapes

:	2*
dtype02
layer-1/MatMul/ReadVariableOp?
layer-1/MatMulMatMulproject/sub:z:0%layer-1/MatMul/ReadVariableOp:value:0*
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
IdentityIdentityoutput/Softmax:softmax:0^layer-1/BiasAdd/ReadVariableOp^layer-1/MatMul/ReadVariableOp^output/BiasAdd/ReadVariableOp^output/MatMul/ReadVariableOp ^project/matmul_1/ReadVariableOp(^project/matrix_transpose/ReadVariableOp*
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
output/MatMul/ReadVariableOpoutput/MatMul/ReadVariableOp2B
project/matmul_1/ReadVariableOpproject/matmul_1/ReadVariableOp2R
'project/matrix_transpose/ReadVariableOp'project/matrix_transpose/ReadVariableOp:V R
'
_output_shapes
:?????????	
'
_user_specified_nameproject_input
?
?
1__inference_classifier_graph_layer_call_fn_633074
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
GPU 2J 8? *U
fPRN
L__inference_classifier_graph_layer_call_and_return_conditional_losses_6326502
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
?	
?
A__inference_model_layer_call_and_return_conditional_losses_632783

inputs
classifier_graph_632771
classifier_graph_632773
classifier_graph_632775
classifier_graph_632777
classifier_graph_632779
identity??(classifier_graph/StatefulPartitionedCall?
(classifier_graph/StatefulPartitionedCallStatefulPartitionedCallinputsclassifier_graph_632771classifier_graph_632773classifier_graph_632775classifier_graph_632777classifier_graph_632779*
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
GPU 2J 8? *U
fPRN
L__inference_classifier_graph_layer_call_and_return_conditional_losses_6327072*
(classifier_graph/StatefulPartitionedCall?
IdentityIdentity1classifier_graph/StatefulPartitionedCall:output:0)^classifier_graph/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????	:::::2T
(classifier_graph/StatefulPartitionedCall(classifier_graph/StatefulPartitionedCall:O K
'
_output_shapes
:?????????	
 
_user_specified_nameinputs
?'
?
L__inference_classifier_graph_layer_call_and_return_conditional_losses_632707
x?
;sequential_project_matrix_transpose_readvariableop_resource5
1sequential_layer_1_matmul_readvariableop_resource6
2sequential_layer_1_biasadd_readvariableop_resource4
0sequential_output_matmul_readvariableop_resource5
1sequential_output_biasadd_readvariableop_resource
identity??)sequential/layer-1/BiasAdd/ReadVariableOp?(sequential/layer-1/MatMul/ReadVariableOp?(sequential/output/BiasAdd/ReadVariableOp?'sequential/output/MatMul/ReadVariableOp?*sequential/project/matmul_1/ReadVariableOp?2sequential/project/matrix_transpose/ReadVariableOp?
2sequential/project/matrix_transpose/ReadVariableOpReadVariableOp;sequential_project_matrix_transpose_readvariableop_resource*
_output_shapes

:	*
dtype024
2sequential/project/matrix_transpose/ReadVariableOp?
2sequential/project/matrix_transpose/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       24
2sequential/project/matrix_transpose/transpose/perm?
-sequential/project/matrix_transpose/transpose	Transpose:sequential/project/matrix_transpose/ReadVariableOp:value:0;sequential/project/matrix_transpose/transpose/perm:output:0*
T0*
_output_shapes

:	2/
-sequential/project/matrix_transpose/transpose?
sequential/project/matmulMatMulx1sequential/project/matrix_transpose/transpose:y:0*
T0*'
_output_shapes
:?????????2
sequential/project/matmul?
*sequential/project/matmul_1/ReadVariableOpReadVariableOp;sequential_project_matrix_transpose_readvariableop_resource*
_output_shapes

:	*
dtype02,
*sequential/project/matmul_1/ReadVariableOp?
sequential/project/matmul_1MatMul#sequential/project/matmul:product:02sequential/project/matmul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	2
sequential/project/matmul_1?
sequential/project/subSubx%sequential/project/matmul_1:product:0*
T0*'
_output_shapes
:?????????	2
sequential/project/sub?
(sequential/layer-1/MatMul/ReadVariableOpReadVariableOp1sequential_layer_1_matmul_readvariableop_resource*
_output_shapes

:	2*
dtype02*
(sequential/layer-1/MatMul/ReadVariableOp?
sequential/layer-1/MatMulMatMulsequential/project/sub:z:00sequential/layer-1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
sequential/layer-1/MatMul?
)sequential/layer-1/BiasAdd/ReadVariableOpReadVariableOp2sequential_layer_1_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02+
)sequential/layer-1/BiasAdd/ReadVariableOp?
sequential/layer-1/BiasAddBiasAdd#sequential/layer-1/MatMul:product:01sequential/layer-1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
sequential/layer-1/BiasAdd?
sequential/layer-1/ReluRelu#sequential/layer-1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????22
sequential/layer-1/Relu?
'sequential/output/MatMul/ReadVariableOpReadVariableOp0sequential_output_matmul_readvariableop_resource*
_output_shapes

:2*
dtype02)
'sequential/output/MatMul/ReadVariableOp?
sequential/output/MatMulMatMul%sequential/layer-1/Relu:activations:0/sequential/output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential/output/MatMul?
(sequential/output/BiasAdd/ReadVariableOpReadVariableOp1sequential_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(sequential/output/BiasAdd/ReadVariableOp?
sequential/output/BiasAddBiasAdd"sequential/output/MatMul:product:00sequential/output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential/output/BiasAdd?
sequential/output/SoftmaxSoftmax"sequential/output/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
sequential/output/Softmax?
IdentityIdentity#sequential/output/Softmax:softmax:0*^sequential/layer-1/BiasAdd/ReadVariableOp)^sequential/layer-1/MatMul/ReadVariableOp)^sequential/output/BiasAdd/ReadVariableOp(^sequential/output/MatMul/ReadVariableOp+^sequential/project/matmul_1/ReadVariableOp3^sequential/project/matrix_transpose/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????	:::::2V
)sequential/layer-1/BiasAdd/ReadVariableOp)sequential/layer-1/BiasAdd/ReadVariableOp2T
(sequential/layer-1/MatMul/ReadVariableOp(sequential/layer-1/MatMul/ReadVariableOp2T
(sequential/output/BiasAdd/ReadVariableOp(sequential/output/BiasAdd/ReadVariableOp2R
'sequential/output/MatMul/ReadVariableOp'sequential/output/MatMul/ReadVariableOp2X
*sequential/project/matmul_1/ReadVariableOp*sequential/project/matmul_1/ReadVariableOp2h
2sequential/project/matrix_transpose/ReadVariableOp2sequential/project/matrix_transpose/ReadVariableOp:J F
'
_output_shapes
:?????????	

_user_specified_namex
?
?
+__inference_sequential_layer_call_fn_633156
project_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallproject_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3*
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
GPU 2J 8? *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_6325262
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????	:::::22
StatefulPartitionedCallStatefulPartitionedCall:V R
'
_output_shapes
:?????????	
'
_user_specified_nameproject_input"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
;
input_10
serving_default_input_1:0?????????	D
classifier_graph0
StatefulPartitionedCall:0?????????tensorflow/serving/predict:ݡ
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
_tf_keras_network?{"class_name": "Functional", "name": "model", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 9]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "ClassifierGraph", "config": {"layer was saved without config": true}, "name": "classifier_graph", "inbound_nodes": [[["input_1", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["classifier_graph", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 9]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 9]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional"}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input_1", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 9]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 9]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}}
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
_tf_keras_model?{"class_name": "ClassifierGraph", "name": "classifier_graph", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "is_graph_network": false, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "ClassifierGraph"}}
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
_tf_keras_sequential?{"class_name": "Sequential", "name": "sequential", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 9]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "project_input"}}, {"class_name": "Project", "config": {"layer was saved without config": true}}, {"class_name": "Dense", "config": {"name": "layer-1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 9]}, "dtype": "float32", "units": 50, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "output", "trainable": true, "dtype": "float32", "units": 2, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 9]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 9]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential"}}
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
_tf_keras_layer?{"class_name": "Project", "name": "project", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 9]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}}
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
!__inference__wrapped_model_632406?
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
input_1?????????	
?2?
A__inference_model_layer_call_and_return_conditional_losses_632750
A__inference_model_layer_call_and_return_conditional_losses_632895
A__inference_model_layer_call_and_return_conditional_losses_632869
A__inference_model_layer_call_and_return_conditional_losses_632765?
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
&__inference_model_layer_call_fn_632796
&__inference_model_layer_call_fn_632910
&__inference_model_layer_call_fn_632826
&__inference_model_layer_call_fn_632925?
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
L__inference_classifier_graph_layer_call_and_return_conditional_losses_633059
L__inference_classifier_graph_layer_call_and_return_conditional_losses_633033
L__inference_classifier_graph_layer_call_and_return_conditional_losses_632977
L__inference_classifier_graph_layer_call_and_return_conditional_losses_632951?
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
1__inference_classifier_graph_layer_call_fn_633007
1__inference_classifier_graph_layer_call_fn_633089
1__inference_classifier_graph_layer_call_fn_633074
1__inference_classifier_graph_layer_call_fn_632992?
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
$__inference_signature_wrapper_632843input_1"?
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
F__inference_sequential_layer_call_and_return_conditional_losses_633223
F__inference_sequential_layer_call_and_return_conditional_losses_633197
F__inference_sequential_layer_call_and_return_conditional_losses_633141
F__inference_sequential_layer_call_and_return_conditional_losses_633115?
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
+__inference_sequential_layer_call_fn_633171
+__inference_sequential_layer_call_fn_633156
+__inference_sequential_layer_call_fn_633238
+__inference_sequential_layer_call_fn_633253?
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
C__inference_project_layer_call_and_return_conditional_losses_632419?
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
(__inference_project_layer_call_fn_632427?
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
C__inference_layer-1_layer_call_and_return_conditional_losses_633264?
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
(__inference_layer-1_layer_call_fn_633273?
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
B__inference_output_layer_call_and_return_conditional_losses_633284?
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
'__inference_output_layer_call_fn_633293?
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
!__inference__wrapped_model_632406~0?-
&?#
!?
input_1?????????	
? "C?@
>
classifier_graph*?'
classifier_graph??????????
L__inference_classifier_graph_layer_call_and_return_conditional_losses_632951h8?5
.?+
!?
input_1?????????	
p 
p
? "%?"
?
0?????????
? ?
L__inference_classifier_graph_layer_call_and_return_conditional_losses_632977h8?5
.?+
!?
input_1?????????	
p 
p 
? "%?"
?
0?????????
? ?
L__inference_classifier_graph_layer_call_and_return_conditional_losses_633033b2?/
(?%
?
x?????????	
p 
p
? "%?"
?
0?????????
? ?
L__inference_classifier_graph_layer_call_and_return_conditional_losses_633059b2?/
(?%
?
x?????????	
p 
p 
? "%?"
?
0?????????
? ?
1__inference_classifier_graph_layer_call_fn_632992[8?5
.?+
!?
input_1?????????	
p 
p
? "???????????
1__inference_classifier_graph_layer_call_fn_633007[8?5
.?+
!?
input_1?????????	
p 
p 
? "???????????
1__inference_classifier_graph_layer_call_fn_633074U2?/
(?%
?
x?????????	
p 
p
? "???????????
1__inference_classifier_graph_layer_call_fn_633089U2?/
(?%
?
x?????????	
p 
p 
? "???????????
C__inference_layer-1_layer_call_and_return_conditional_losses_633264\/?,
%?"
 ?
inputs?????????	
? "%?"
?
0?????????2
? {
(__inference_layer-1_layer_call_fn_633273O/?,
%?"
 ?
inputs?????????	
? "??????????2?
A__inference_model_layer_call_and_return_conditional_losses_632750h8?5
.?+
!?
input_1?????????	
p

 
? "%?"
?
0?????????
? ?
A__inference_model_layer_call_and_return_conditional_losses_632765h8?5
.?+
!?
input_1?????????	
p 

 
? "%?"
?
0?????????
? ?
A__inference_model_layer_call_and_return_conditional_losses_632869g7?4
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
A__inference_model_layer_call_and_return_conditional_losses_632895g7?4
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
&__inference_model_layer_call_fn_632796[8?5
.?+
!?
input_1?????????	
p

 
? "???????????
&__inference_model_layer_call_fn_632826[8?5
.?+
!?
input_1?????????	
p 

 
? "???????????
&__inference_model_layer_call_fn_632910Z7?4
-?*
 ?
inputs?????????	
p

 
? "???????????
&__inference_model_layer_call_fn_632925Z7?4
-?*
 ?
inputs?????????	
p 

 
? "???????????
B__inference_output_layer_call_and_return_conditional_losses_633284\/?,
%?"
 ?
inputs?????????2
? "%?"
?
0?????????
? z
'__inference_output_layer_call_fn_633293O/?,
%?"
 ?
inputs?????????2
? "???????????
C__inference_project_layer_call_and_return_conditional_losses_632419V*?'
 ?
?
x?????????	
? "%?"
?
0?????????	
? u
(__inference_project_layer_call_fn_632427I*?'
 ?
?
x?????????	
? "??????????	?
F__inference_sequential_layer_call_and_return_conditional_losses_633115n>?;
4?1
'?$
project_input?????????	
p

 
? "%?"
?
0?????????
? ?
F__inference_sequential_layer_call_and_return_conditional_losses_633141n>?;
4?1
'?$
project_input?????????	
p 

 
? "%?"
?
0?????????
? ?
F__inference_sequential_layer_call_and_return_conditional_losses_633197g7?4
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
F__inference_sequential_layer_call_and_return_conditional_losses_633223g7?4
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
+__inference_sequential_layer_call_fn_633156a>?;
4?1
'?$
project_input?????????	
p

 
? "???????????
+__inference_sequential_layer_call_fn_633171a>?;
4?1
'?$
project_input?????????	
p 

 
? "???????????
+__inference_sequential_layer_call_fn_633238Z7?4
-?*
 ?
inputs?????????	
p

 
? "???????????
+__inference_sequential_layer_call_fn_633253Z7?4
-?*
 ?
inputs?????????	
p 

 
? "???????????
$__inference_signature_wrapper_632843?;?8
? 
1?.
,
input_1!?
input_1?????????	"C?@
>
classifier_graph*?'
classifier_graph?????????