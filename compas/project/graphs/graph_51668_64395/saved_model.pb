ʐ
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
{
serving_default_input_10Placeholder*'
_output_shapes
:?????????	*
dtype0*
shape:?????????	
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_10Variablelayer-1/kernellayer-1/biasoutput/kerneloutput/bias*
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
%__inference_signature_wrapper_6333344
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
 __inference__traced_save_6333832
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
#__inference__traced_restore_6333857??
?
?
I__inference_sequential_9_layer_call_and_return_conditional_losses_6333059

inputs
project_9_6333045
layer_1_6333048
layer_1_6333050
output_6333053
output_6333055
identity??layer-1/StatefulPartitionedCall?output/StatefulPartitionedCall?!project_9/StatefulPartitionedCall?
!project_9/StatefulPartitionedCallStatefulPartitionedCallinputsproject_9_6333045*
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
F__inference_project_9_layer_call_and_return_conditional_losses_63329202#
!project_9/StatefulPartitionedCall?
layer-1/StatefulPartitionedCallStatefulPartitionedCall*project_9/StatefulPartitionedCall:output:0layer_1_6333048layer_1_6333050*
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
D__inference_layer-1_layer_call_and_return_conditional_losses_63329462!
layer-1/StatefulPartitionedCall?
output/StatefulPartitionedCallStatefulPartitionedCall(layer-1/StatefulPartitionedCall:output:0output_6333053output_6333055*
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
C__inference_output_layer_call_and_return_conditional_losses_63329732 
output/StatefulPartitionedCall?
IdentityIdentity'output/StatefulPartitionedCall:output:0 ^layer-1/StatefulPartitionedCall^output/StatefulPartitionedCall"^project_9/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????	:::::2B
layer-1/StatefulPartitionedCalllayer-1/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall2F
!project_9/StatefulPartitionedCall!project_9/StatefulPartitionedCall:O K
'
_output_shapes
:?????????	
 
_user_specified_nameinputs
?
}
(__inference_output_layer_call_fn_6333794

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
C__inference_output_layer_call_and_return_conditional_losses_63329732
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
?

?
D__inference_model_9_layer_call_and_return_conditional_losses_6333251
input_10
classifier_graph_9_6333239
classifier_graph_9_6333241
classifier_graph_9_6333243
classifier_graph_9_6333245
classifier_graph_9_6333247
identity??*classifier_graph_9/StatefulPartitionedCall?
*classifier_graph_9/StatefulPartitionedCallStatefulPartitionedCallinput_10classifier_graph_9_6333239classifier_graph_9_6333241classifier_graph_9_6333243classifier_graph_9_6333245classifier_graph_9_6333247*
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
O__inference_classifier_graph_9_layer_call_and_return_conditional_losses_63332082,
*classifier_graph_9/StatefulPartitionedCall?
IdentityIdentity3classifier_graph_9/StatefulPartitionedCall:output:0+^classifier_graph_9/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????	:::::2X
*classifier_graph_9/StatefulPartitionedCall*classifier_graph_9/StatefulPartitionedCall:Q M
'
_output_shapes
:?????????	
"
_user_specified_name
input_10
?!
?
I__inference_sequential_9_layer_call_and_return_conditional_losses_6333616

inputs6
2project_9_matrix_transpose_readvariableop_resource*
&layer_1_matmul_readvariableop_resource+
'layer_1_biasadd_readvariableop_resource)
%output_matmul_readvariableop_resource*
&output_biasadd_readvariableop_resource
identity??layer-1/BiasAdd/ReadVariableOp?layer-1/MatMul/ReadVariableOp?output/BiasAdd/ReadVariableOp?output/MatMul/ReadVariableOp?!project_9/matmul_1/ReadVariableOp?)project_9/matrix_transpose/ReadVariableOp?
)project_9/matrix_transpose/ReadVariableOpReadVariableOp2project_9_matrix_transpose_readvariableop_resource*
_output_shapes

:	*
dtype02+
)project_9/matrix_transpose/ReadVariableOp?
)project_9/matrix_transpose/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2+
)project_9/matrix_transpose/transpose/perm?
$project_9/matrix_transpose/transpose	Transpose1project_9/matrix_transpose/ReadVariableOp:value:02project_9/matrix_transpose/transpose/perm:output:0*
T0*
_output_shapes

:	2&
$project_9/matrix_transpose/transpose?
project_9/matmulMatMulinputs(project_9/matrix_transpose/transpose:y:0*
T0*'
_output_shapes
:?????????2
project_9/matmul?
!project_9/matmul_1/ReadVariableOpReadVariableOp2project_9_matrix_transpose_readvariableop_resource*
_output_shapes

:	*
dtype02#
!project_9/matmul_1/ReadVariableOp?
project_9/matmul_1MatMulproject_9/matmul:product:0)project_9/matmul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	2
project_9/matmul_1}
project_9/subSubinputsproject_9/matmul_1:product:0*
T0*'
_output_shapes
:?????????	2
project_9/sub?
layer-1/MatMul/ReadVariableOpReadVariableOp&layer_1_matmul_readvariableop_resource*
_output_shapes

:	2*
dtype02
layer-1/MatMul/ReadVariableOp?
layer-1/MatMulMatMulproject_9/sub:z:0%layer-1/MatMul/ReadVariableOp:value:0*
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
IdentityIdentityoutput/Softmax:softmax:0^layer-1/BiasAdd/ReadVariableOp^layer-1/MatMul/ReadVariableOp^output/BiasAdd/ReadVariableOp^output/MatMul/ReadVariableOp"^project_9/matmul_1/ReadVariableOp*^project_9/matrix_transpose/ReadVariableOp*
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
!project_9/matmul_1/ReadVariableOp!project_9/matmul_1/ReadVariableOp2V
)project_9/matrix_transpose/ReadVariableOp)project_9/matrix_transpose/ReadVariableOp:O K
'
_output_shapes
:?????????	
 
_user_specified_nameinputs
?
?
.__inference_sequential_9_layer_call_fn_6333754
project_9_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallproject_9_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3*
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
I__inference_sequential_9_layer_call_and_return_conditional_losses_63330592
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
_user_specified_nameproject_9_input
?
~
)__inference_layer-1_layer_call_fn_6333774

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
D__inference_layer-1_layer_call_and_return_conditional_losses_63329462
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
?
?
%__inference_signature_wrapper_6333344
input_10
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_10unknown	unknown_0	unknown_1	unknown_2	unknown_3*
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
"__inference__wrapped_model_63329072
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????	:::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:?????????	
"
_user_specified_name
input_10
?)
?
O__inference_classifier_graph_9_layer_call_and_return_conditional_losses_6333208
xC
?sequential_9_project_9_matrix_transpose_readvariableop_resource7
3sequential_9_layer_1_matmul_readvariableop_resource8
4sequential_9_layer_1_biasadd_readvariableop_resource6
2sequential_9_output_matmul_readvariableop_resource7
3sequential_9_output_biasadd_readvariableop_resource
identity??+sequential_9/layer-1/BiasAdd/ReadVariableOp?*sequential_9/layer-1/MatMul/ReadVariableOp?*sequential_9/output/BiasAdd/ReadVariableOp?)sequential_9/output/MatMul/ReadVariableOp?.sequential_9/project_9/matmul_1/ReadVariableOp?6sequential_9/project_9/matrix_transpose/ReadVariableOp?
6sequential_9/project_9/matrix_transpose/ReadVariableOpReadVariableOp?sequential_9_project_9_matrix_transpose_readvariableop_resource*
_output_shapes

:	*
dtype028
6sequential_9/project_9/matrix_transpose/ReadVariableOp?
6sequential_9/project_9/matrix_transpose/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       28
6sequential_9/project_9/matrix_transpose/transpose/perm?
1sequential_9/project_9/matrix_transpose/transpose	Transpose>sequential_9/project_9/matrix_transpose/ReadVariableOp:value:0?sequential_9/project_9/matrix_transpose/transpose/perm:output:0*
T0*
_output_shapes

:	23
1sequential_9/project_9/matrix_transpose/transpose?
sequential_9/project_9/matmulMatMulx5sequential_9/project_9/matrix_transpose/transpose:y:0*
T0*'
_output_shapes
:?????????2
sequential_9/project_9/matmul?
.sequential_9/project_9/matmul_1/ReadVariableOpReadVariableOp?sequential_9_project_9_matrix_transpose_readvariableop_resource*
_output_shapes

:	*
dtype020
.sequential_9/project_9/matmul_1/ReadVariableOp?
sequential_9/project_9/matmul_1MatMul'sequential_9/project_9/matmul:product:06sequential_9/project_9/matmul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	2!
sequential_9/project_9/matmul_1?
sequential_9/project_9/subSubx)sequential_9/project_9/matmul_1:product:0*
T0*'
_output_shapes
:?????????	2
sequential_9/project_9/sub?
*sequential_9/layer-1/MatMul/ReadVariableOpReadVariableOp3sequential_9_layer_1_matmul_readvariableop_resource*
_output_shapes

:	2*
dtype02,
*sequential_9/layer-1/MatMul/ReadVariableOp?
sequential_9/layer-1/MatMulMatMulsequential_9/project_9/sub:z:02sequential_9/layer-1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
sequential_9/layer-1/MatMul?
+sequential_9/layer-1/BiasAdd/ReadVariableOpReadVariableOp4sequential_9_layer_1_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02-
+sequential_9/layer-1/BiasAdd/ReadVariableOp?
sequential_9/layer-1/BiasAddBiasAdd%sequential_9/layer-1/MatMul:product:03sequential_9/layer-1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
sequential_9/layer-1/BiasAdd?
sequential_9/layer-1/ReluRelu%sequential_9/layer-1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????22
sequential_9/layer-1/Relu?
)sequential_9/output/MatMul/ReadVariableOpReadVariableOp2sequential_9_output_matmul_readvariableop_resource*
_output_shapes

:2*
dtype02+
)sequential_9/output/MatMul/ReadVariableOp?
sequential_9/output/MatMulMatMul'sequential_9/layer-1/Relu:activations:01sequential_9/output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential_9/output/MatMul?
*sequential_9/output/BiasAdd/ReadVariableOpReadVariableOp3sequential_9_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02,
*sequential_9/output/BiasAdd/ReadVariableOp?
sequential_9/output/BiasAddBiasAdd$sequential_9/output/MatMul:product:02sequential_9/output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential_9/output/BiasAdd?
sequential_9/output/SoftmaxSoftmax$sequential_9/output/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
sequential_9/output/Softmax?
IdentityIdentity%sequential_9/output/Softmax:softmax:0,^sequential_9/layer-1/BiasAdd/ReadVariableOp+^sequential_9/layer-1/MatMul/ReadVariableOp+^sequential_9/output/BiasAdd/ReadVariableOp*^sequential_9/output/MatMul/ReadVariableOp/^sequential_9/project_9/matmul_1/ReadVariableOp7^sequential_9/project_9/matrix_transpose/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????	:::::2Z
+sequential_9/layer-1/BiasAdd/ReadVariableOp+sequential_9/layer-1/BiasAdd/ReadVariableOp2X
*sequential_9/layer-1/MatMul/ReadVariableOp*sequential_9/layer-1/MatMul/ReadVariableOp2X
*sequential_9/output/BiasAdd/ReadVariableOp*sequential_9/output/BiasAdd/ReadVariableOp2V
)sequential_9/output/MatMul/ReadVariableOp)sequential_9/output/MatMul/ReadVariableOp2`
.sequential_9/project_9/matmul_1/ReadVariableOp.sequential_9/project_9/matmul_1/ReadVariableOp2p
6sequential_9/project_9/matrix_transpose/ReadVariableOp6sequential_9/project_9/matrix_transpose/ReadVariableOp:J F
'
_output_shapes
:?????????	

_user_specified_namex
?

?
D__inference_model_9_layer_call_and_return_conditional_losses_6333314

inputs
classifier_graph_9_6333302
classifier_graph_9_6333304
classifier_graph_9_6333306
classifier_graph_9_6333308
classifier_graph_9_6333310
identity??*classifier_graph_9/StatefulPartitionedCall?
*classifier_graph_9/StatefulPartitionedCallStatefulPartitionedCallinputsclassifier_graph_9_6333302classifier_graph_9_6333304classifier_graph_9_6333306classifier_graph_9_6333308classifier_graph_9_6333310*
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
O__inference_classifier_graph_9_layer_call_and_return_conditional_losses_63331512,
*classifier_graph_9/StatefulPartitionedCall?
IdentityIdentity3classifier_graph_9/StatefulPartitionedCall:output:0+^classifier_graph_9/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????	:::::2X
*classifier_graph_9/StatefulPartitionedCall*classifier_graph_9/StatefulPartitionedCall:O K
'
_output_shapes
:?????????	
 
_user_specified_nameinputs
?
l
+__inference_project_9_layer_call_fn_6332928
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
F__inference_project_9_layer_call_and_return_conditional_losses_63329202
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
O__inference_classifier_graph_9_layer_call_and_return_conditional_losses_6333478
input_1C
?sequential_9_project_9_matrix_transpose_readvariableop_resource7
3sequential_9_layer_1_matmul_readvariableop_resource8
4sequential_9_layer_1_biasadd_readvariableop_resource6
2sequential_9_output_matmul_readvariableop_resource7
3sequential_9_output_biasadd_readvariableop_resource
identity??+sequential_9/layer-1/BiasAdd/ReadVariableOp?*sequential_9/layer-1/MatMul/ReadVariableOp?*sequential_9/output/BiasAdd/ReadVariableOp?)sequential_9/output/MatMul/ReadVariableOp?.sequential_9/project_9/matmul_1/ReadVariableOp?6sequential_9/project_9/matrix_transpose/ReadVariableOp?
6sequential_9/project_9/matrix_transpose/ReadVariableOpReadVariableOp?sequential_9_project_9_matrix_transpose_readvariableop_resource*
_output_shapes

:	*
dtype028
6sequential_9/project_9/matrix_transpose/ReadVariableOp?
6sequential_9/project_9/matrix_transpose/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       28
6sequential_9/project_9/matrix_transpose/transpose/perm?
1sequential_9/project_9/matrix_transpose/transpose	Transpose>sequential_9/project_9/matrix_transpose/ReadVariableOp:value:0?sequential_9/project_9/matrix_transpose/transpose/perm:output:0*
T0*
_output_shapes

:	23
1sequential_9/project_9/matrix_transpose/transpose?
sequential_9/project_9/matmulMatMulinput_15sequential_9/project_9/matrix_transpose/transpose:y:0*
T0*'
_output_shapes
:?????????2
sequential_9/project_9/matmul?
.sequential_9/project_9/matmul_1/ReadVariableOpReadVariableOp?sequential_9_project_9_matrix_transpose_readvariableop_resource*
_output_shapes

:	*
dtype020
.sequential_9/project_9/matmul_1/ReadVariableOp?
sequential_9/project_9/matmul_1MatMul'sequential_9/project_9/matmul:product:06sequential_9/project_9/matmul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	2!
sequential_9/project_9/matmul_1?
sequential_9/project_9/subSubinput_1)sequential_9/project_9/matmul_1:product:0*
T0*'
_output_shapes
:?????????	2
sequential_9/project_9/sub?
*sequential_9/layer-1/MatMul/ReadVariableOpReadVariableOp3sequential_9_layer_1_matmul_readvariableop_resource*
_output_shapes

:	2*
dtype02,
*sequential_9/layer-1/MatMul/ReadVariableOp?
sequential_9/layer-1/MatMulMatMulsequential_9/project_9/sub:z:02sequential_9/layer-1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
sequential_9/layer-1/MatMul?
+sequential_9/layer-1/BiasAdd/ReadVariableOpReadVariableOp4sequential_9_layer_1_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02-
+sequential_9/layer-1/BiasAdd/ReadVariableOp?
sequential_9/layer-1/BiasAddBiasAdd%sequential_9/layer-1/MatMul:product:03sequential_9/layer-1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
sequential_9/layer-1/BiasAdd?
sequential_9/layer-1/ReluRelu%sequential_9/layer-1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????22
sequential_9/layer-1/Relu?
)sequential_9/output/MatMul/ReadVariableOpReadVariableOp2sequential_9_output_matmul_readvariableop_resource*
_output_shapes

:2*
dtype02+
)sequential_9/output/MatMul/ReadVariableOp?
sequential_9/output/MatMulMatMul'sequential_9/layer-1/Relu:activations:01sequential_9/output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential_9/output/MatMul?
*sequential_9/output/BiasAdd/ReadVariableOpReadVariableOp3sequential_9_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02,
*sequential_9/output/BiasAdd/ReadVariableOp?
sequential_9/output/BiasAddBiasAdd$sequential_9/output/MatMul:product:02sequential_9/output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential_9/output/BiasAdd?
sequential_9/output/SoftmaxSoftmax$sequential_9/output/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
sequential_9/output/Softmax?
IdentityIdentity%sequential_9/output/Softmax:softmax:0,^sequential_9/layer-1/BiasAdd/ReadVariableOp+^sequential_9/layer-1/MatMul/ReadVariableOp+^sequential_9/output/BiasAdd/ReadVariableOp*^sequential_9/output/MatMul/ReadVariableOp/^sequential_9/project_9/matmul_1/ReadVariableOp7^sequential_9/project_9/matrix_transpose/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????	:::::2Z
+sequential_9/layer-1/BiasAdd/ReadVariableOp+sequential_9/layer-1/BiasAdd/ReadVariableOp2X
*sequential_9/layer-1/MatMul/ReadVariableOp*sequential_9/layer-1/MatMul/ReadVariableOp2X
*sequential_9/output/BiasAdd/ReadVariableOp*sequential_9/output/BiasAdd/ReadVariableOp2V
)sequential_9/output/MatMul/ReadVariableOp)sequential_9/output/MatMul/ReadVariableOp2`
.sequential_9/project_9/matmul_1/ReadVariableOp.sequential_9/project_9/matmul_1/ReadVariableOp2p
6sequential_9/project_9/matrix_transpose/ReadVariableOp6sequential_9/project_9/matrix_transpose/ReadVariableOp:P L
'
_output_shapes
:?????????	
!
_user_specified_name	input_1
?	
?
C__inference_output_layer_call_and_return_conditional_losses_6333785

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
?	
?
D__inference_layer-1_layer_call_and_return_conditional_losses_6333765

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
4__inference_classifier_graph_9_layer_call_fn_6333590
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
O__inference_classifier_graph_9_layer_call_and_return_conditional_losses_63331512
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
?;
?
"__inference__wrapped_model_6332907
input_10^
Zmodel_9_classifier_graph_9_sequential_9_project_9_matrix_transpose_readvariableop_resourceR
Nmodel_9_classifier_graph_9_sequential_9_layer_1_matmul_readvariableop_resourceS
Omodel_9_classifier_graph_9_sequential_9_layer_1_biasadd_readvariableop_resourceQ
Mmodel_9_classifier_graph_9_sequential_9_output_matmul_readvariableop_resourceR
Nmodel_9_classifier_graph_9_sequential_9_output_biasadd_readvariableop_resource
identity??Fmodel_9/classifier_graph_9/sequential_9/layer-1/BiasAdd/ReadVariableOp?Emodel_9/classifier_graph_9/sequential_9/layer-1/MatMul/ReadVariableOp?Emodel_9/classifier_graph_9/sequential_9/output/BiasAdd/ReadVariableOp?Dmodel_9/classifier_graph_9/sequential_9/output/MatMul/ReadVariableOp?Imodel_9/classifier_graph_9/sequential_9/project_9/matmul_1/ReadVariableOp?Qmodel_9/classifier_graph_9/sequential_9/project_9/matrix_transpose/ReadVariableOp?
Qmodel_9/classifier_graph_9/sequential_9/project_9/matrix_transpose/ReadVariableOpReadVariableOpZmodel_9_classifier_graph_9_sequential_9_project_9_matrix_transpose_readvariableop_resource*
_output_shapes

:	*
dtype02S
Qmodel_9/classifier_graph_9/sequential_9/project_9/matrix_transpose/ReadVariableOp?
Qmodel_9/classifier_graph_9/sequential_9/project_9/matrix_transpose/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2S
Qmodel_9/classifier_graph_9/sequential_9/project_9/matrix_transpose/transpose/perm?
Lmodel_9/classifier_graph_9/sequential_9/project_9/matrix_transpose/transpose	TransposeYmodel_9/classifier_graph_9/sequential_9/project_9/matrix_transpose/ReadVariableOp:value:0Zmodel_9/classifier_graph_9/sequential_9/project_9/matrix_transpose/transpose/perm:output:0*
T0*
_output_shapes

:	2N
Lmodel_9/classifier_graph_9/sequential_9/project_9/matrix_transpose/transpose?
8model_9/classifier_graph_9/sequential_9/project_9/matmulMatMulinput_10Pmodel_9/classifier_graph_9/sequential_9/project_9/matrix_transpose/transpose:y:0*
T0*'
_output_shapes
:?????????2:
8model_9/classifier_graph_9/sequential_9/project_9/matmul?
Imodel_9/classifier_graph_9/sequential_9/project_9/matmul_1/ReadVariableOpReadVariableOpZmodel_9_classifier_graph_9_sequential_9_project_9_matrix_transpose_readvariableop_resource*
_output_shapes

:	*
dtype02K
Imodel_9/classifier_graph_9/sequential_9/project_9/matmul_1/ReadVariableOp?
:model_9/classifier_graph_9/sequential_9/project_9/matmul_1MatMulBmodel_9/classifier_graph_9/sequential_9/project_9/matmul:product:0Qmodel_9/classifier_graph_9/sequential_9/project_9/matmul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	2<
:model_9/classifier_graph_9/sequential_9/project_9/matmul_1?
5model_9/classifier_graph_9/sequential_9/project_9/subSubinput_10Dmodel_9/classifier_graph_9/sequential_9/project_9/matmul_1:product:0*
T0*'
_output_shapes
:?????????	27
5model_9/classifier_graph_9/sequential_9/project_9/sub?
Emodel_9/classifier_graph_9/sequential_9/layer-1/MatMul/ReadVariableOpReadVariableOpNmodel_9_classifier_graph_9_sequential_9_layer_1_matmul_readvariableop_resource*
_output_shapes

:	2*
dtype02G
Emodel_9/classifier_graph_9/sequential_9/layer-1/MatMul/ReadVariableOp?
6model_9/classifier_graph_9/sequential_9/layer-1/MatMulMatMul9model_9/classifier_graph_9/sequential_9/project_9/sub:z:0Mmodel_9/classifier_graph_9/sequential_9/layer-1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????228
6model_9/classifier_graph_9/sequential_9/layer-1/MatMul?
Fmodel_9/classifier_graph_9/sequential_9/layer-1/BiasAdd/ReadVariableOpReadVariableOpOmodel_9_classifier_graph_9_sequential_9_layer_1_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02H
Fmodel_9/classifier_graph_9/sequential_9/layer-1/BiasAdd/ReadVariableOp?
7model_9/classifier_graph_9/sequential_9/layer-1/BiasAddBiasAdd@model_9/classifier_graph_9/sequential_9/layer-1/MatMul:product:0Nmodel_9/classifier_graph_9/sequential_9/layer-1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????229
7model_9/classifier_graph_9/sequential_9/layer-1/BiasAdd?
4model_9/classifier_graph_9/sequential_9/layer-1/ReluRelu@model_9/classifier_graph_9/sequential_9/layer-1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????226
4model_9/classifier_graph_9/sequential_9/layer-1/Relu?
Dmodel_9/classifier_graph_9/sequential_9/output/MatMul/ReadVariableOpReadVariableOpMmodel_9_classifier_graph_9_sequential_9_output_matmul_readvariableop_resource*
_output_shapes

:2*
dtype02F
Dmodel_9/classifier_graph_9/sequential_9/output/MatMul/ReadVariableOp?
5model_9/classifier_graph_9/sequential_9/output/MatMulMatMulBmodel_9/classifier_graph_9/sequential_9/layer-1/Relu:activations:0Lmodel_9/classifier_graph_9/sequential_9/output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????27
5model_9/classifier_graph_9/sequential_9/output/MatMul?
Emodel_9/classifier_graph_9/sequential_9/output/BiasAdd/ReadVariableOpReadVariableOpNmodel_9_classifier_graph_9_sequential_9_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02G
Emodel_9/classifier_graph_9/sequential_9/output/BiasAdd/ReadVariableOp?
6model_9/classifier_graph_9/sequential_9/output/BiasAddBiasAdd?model_9/classifier_graph_9/sequential_9/output/MatMul:product:0Mmodel_9/classifier_graph_9/sequential_9/output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????28
6model_9/classifier_graph_9/sequential_9/output/BiasAdd?
6model_9/classifier_graph_9/sequential_9/output/SoftmaxSoftmax?model_9/classifier_graph_9/sequential_9/output/BiasAdd:output:0*
T0*'
_output_shapes
:?????????28
6model_9/classifier_graph_9/sequential_9/output/Softmax?
IdentityIdentity@model_9/classifier_graph_9/sequential_9/output/Softmax:softmax:0G^model_9/classifier_graph_9/sequential_9/layer-1/BiasAdd/ReadVariableOpF^model_9/classifier_graph_9/sequential_9/layer-1/MatMul/ReadVariableOpF^model_9/classifier_graph_9/sequential_9/output/BiasAdd/ReadVariableOpE^model_9/classifier_graph_9/sequential_9/output/MatMul/ReadVariableOpJ^model_9/classifier_graph_9/sequential_9/project_9/matmul_1/ReadVariableOpR^model_9/classifier_graph_9/sequential_9/project_9/matrix_transpose/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????	:::::2?
Fmodel_9/classifier_graph_9/sequential_9/layer-1/BiasAdd/ReadVariableOpFmodel_9/classifier_graph_9/sequential_9/layer-1/BiasAdd/ReadVariableOp2?
Emodel_9/classifier_graph_9/sequential_9/layer-1/MatMul/ReadVariableOpEmodel_9/classifier_graph_9/sequential_9/layer-1/MatMul/ReadVariableOp2?
Emodel_9/classifier_graph_9/sequential_9/output/BiasAdd/ReadVariableOpEmodel_9/classifier_graph_9/sequential_9/output/BiasAdd/ReadVariableOp2?
Dmodel_9/classifier_graph_9/sequential_9/output/MatMul/ReadVariableOpDmodel_9/classifier_graph_9/sequential_9/output/MatMul/ReadVariableOp2?
Imodel_9/classifier_graph_9/sequential_9/project_9/matmul_1/ReadVariableOpImodel_9/classifier_graph_9/sequential_9/project_9/matmul_1/ReadVariableOp2?
Qmodel_9/classifier_graph_9/sequential_9/project_9/matrix_transpose/ReadVariableOpQmodel_9/classifier_graph_9/sequential_9/project_9/matrix_transpose/ReadVariableOp:Q M
'
_output_shapes
:?????????	
"
_user_specified_name
input_10
?6
?
D__inference_model_9_layer_call_and_return_conditional_losses_6333396

inputsV
Rclassifier_graph_9_sequential_9_project_9_matrix_transpose_readvariableop_resourceJ
Fclassifier_graph_9_sequential_9_layer_1_matmul_readvariableop_resourceK
Gclassifier_graph_9_sequential_9_layer_1_biasadd_readvariableop_resourceI
Eclassifier_graph_9_sequential_9_output_matmul_readvariableop_resourceJ
Fclassifier_graph_9_sequential_9_output_biasadd_readvariableop_resource
identity??>classifier_graph_9/sequential_9/layer-1/BiasAdd/ReadVariableOp?=classifier_graph_9/sequential_9/layer-1/MatMul/ReadVariableOp?=classifier_graph_9/sequential_9/output/BiasAdd/ReadVariableOp?<classifier_graph_9/sequential_9/output/MatMul/ReadVariableOp?Aclassifier_graph_9/sequential_9/project_9/matmul_1/ReadVariableOp?Iclassifier_graph_9/sequential_9/project_9/matrix_transpose/ReadVariableOp?
Iclassifier_graph_9/sequential_9/project_9/matrix_transpose/ReadVariableOpReadVariableOpRclassifier_graph_9_sequential_9_project_9_matrix_transpose_readvariableop_resource*
_output_shapes

:	*
dtype02K
Iclassifier_graph_9/sequential_9/project_9/matrix_transpose/ReadVariableOp?
Iclassifier_graph_9/sequential_9/project_9/matrix_transpose/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2K
Iclassifier_graph_9/sequential_9/project_9/matrix_transpose/transpose/perm?
Dclassifier_graph_9/sequential_9/project_9/matrix_transpose/transpose	TransposeQclassifier_graph_9/sequential_9/project_9/matrix_transpose/ReadVariableOp:value:0Rclassifier_graph_9/sequential_9/project_9/matrix_transpose/transpose/perm:output:0*
T0*
_output_shapes

:	2F
Dclassifier_graph_9/sequential_9/project_9/matrix_transpose/transpose?
0classifier_graph_9/sequential_9/project_9/matmulMatMulinputsHclassifier_graph_9/sequential_9/project_9/matrix_transpose/transpose:y:0*
T0*'
_output_shapes
:?????????22
0classifier_graph_9/sequential_9/project_9/matmul?
Aclassifier_graph_9/sequential_9/project_9/matmul_1/ReadVariableOpReadVariableOpRclassifier_graph_9_sequential_9_project_9_matrix_transpose_readvariableop_resource*
_output_shapes

:	*
dtype02C
Aclassifier_graph_9/sequential_9/project_9/matmul_1/ReadVariableOp?
2classifier_graph_9/sequential_9/project_9/matmul_1MatMul:classifier_graph_9/sequential_9/project_9/matmul:product:0Iclassifier_graph_9/sequential_9/project_9/matmul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	24
2classifier_graph_9/sequential_9/project_9/matmul_1?
-classifier_graph_9/sequential_9/project_9/subSubinputs<classifier_graph_9/sequential_9/project_9/matmul_1:product:0*
T0*'
_output_shapes
:?????????	2/
-classifier_graph_9/sequential_9/project_9/sub?
=classifier_graph_9/sequential_9/layer-1/MatMul/ReadVariableOpReadVariableOpFclassifier_graph_9_sequential_9_layer_1_matmul_readvariableop_resource*
_output_shapes

:	2*
dtype02?
=classifier_graph_9/sequential_9/layer-1/MatMul/ReadVariableOp?
.classifier_graph_9/sequential_9/layer-1/MatMulMatMul1classifier_graph_9/sequential_9/project_9/sub:z:0Eclassifier_graph_9/sequential_9/layer-1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????220
.classifier_graph_9/sequential_9/layer-1/MatMul?
>classifier_graph_9/sequential_9/layer-1/BiasAdd/ReadVariableOpReadVariableOpGclassifier_graph_9_sequential_9_layer_1_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02@
>classifier_graph_9/sequential_9/layer-1/BiasAdd/ReadVariableOp?
/classifier_graph_9/sequential_9/layer-1/BiasAddBiasAdd8classifier_graph_9/sequential_9/layer-1/MatMul:product:0Fclassifier_graph_9/sequential_9/layer-1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????221
/classifier_graph_9/sequential_9/layer-1/BiasAdd?
,classifier_graph_9/sequential_9/layer-1/ReluRelu8classifier_graph_9/sequential_9/layer-1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????22.
,classifier_graph_9/sequential_9/layer-1/Relu?
<classifier_graph_9/sequential_9/output/MatMul/ReadVariableOpReadVariableOpEclassifier_graph_9_sequential_9_output_matmul_readvariableop_resource*
_output_shapes

:2*
dtype02>
<classifier_graph_9/sequential_9/output/MatMul/ReadVariableOp?
-classifier_graph_9/sequential_9/output/MatMulMatMul:classifier_graph_9/sequential_9/layer-1/Relu:activations:0Dclassifier_graph_9/sequential_9/output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2/
-classifier_graph_9/sequential_9/output/MatMul?
=classifier_graph_9/sequential_9/output/BiasAdd/ReadVariableOpReadVariableOpFclassifier_graph_9_sequential_9_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02?
=classifier_graph_9/sequential_9/output/BiasAdd/ReadVariableOp?
.classifier_graph_9/sequential_9/output/BiasAddBiasAdd7classifier_graph_9/sequential_9/output/MatMul:product:0Eclassifier_graph_9/sequential_9/output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????20
.classifier_graph_9/sequential_9/output/BiasAdd?
.classifier_graph_9/sequential_9/output/SoftmaxSoftmax7classifier_graph_9/sequential_9/output/BiasAdd:output:0*
T0*'
_output_shapes
:?????????20
.classifier_graph_9/sequential_9/output/Softmax?
IdentityIdentity8classifier_graph_9/sequential_9/output/Softmax:softmax:0?^classifier_graph_9/sequential_9/layer-1/BiasAdd/ReadVariableOp>^classifier_graph_9/sequential_9/layer-1/MatMul/ReadVariableOp>^classifier_graph_9/sequential_9/output/BiasAdd/ReadVariableOp=^classifier_graph_9/sequential_9/output/MatMul/ReadVariableOpB^classifier_graph_9/sequential_9/project_9/matmul_1/ReadVariableOpJ^classifier_graph_9/sequential_9/project_9/matrix_transpose/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????	:::::2?
>classifier_graph_9/sequential_9/layer-1/BiasAdd/ReadVariableOp>classifier_graph_9/sequential_9/layer-1/BiasAdd/ReadVariableOp2~
=classifier_graph_9/sequential_9/layer-1/MatMul/ReadVariableOp=classifier_graph_9/sequential_9/layer-1/MatMul/ReadVariableOp2~
=classifier_graph_9/sequential_9/output/BiasAdd/ReadVariableOp=classifier_graph_9/sequential_9/output/BiasAdd/ReadVariableOp2|
<classifier_graph_9/sequential_9/output/MatMul/ReadVariableOp<classifier_graph_9/sequential_9/output/MatMul/ReadVariableOp2?
Aclassifier_graph_9/sequential_9/project_9/matmul_1/ReadVariableOpAclassifier_graph_9/sequential_9/project_9/matmul_1/ReadVariableOp2?
Iclassifier_graph_9/sequential_9/project_9/matrix_transpose/ReadVariableOpIclassifier_graph_9/sequential_9/project_9/matrix_transpose/ReadVariableOp:O K
'
_output_shapes
:?????????	
 
_user_specified_nameinputs
?
?
)__inference_model_9_layer_call_fn_6333411

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
D__inference_model_9_layer_call_and_return_conditional_losses_63332842
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
D__inference_model_9_layer_call_and_return_conditional_losses_6333284

inputs
classifier_graph_9_6333272
classifier_graph_9_6333274
classifier_graph_9_6333276
classifier_graph_9_6333278
classifier_graph_9_6333280
identity??*classifier_graph_9/StatefulPartitionedCall?
*classifier_graph_9/StatefulPartitionedCallStatefulPartitionedCallinputsclassifier_graph_9_6333272classifier_graph_9_6333274classifier_graph_9_6333276classifier_graph_9_6333278classifier_graph_9_6333280*
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
O__inference_classifier_graph_9_layer_call_and_return_conditional_losses_63332082,
*classifier_graph_9/StatefulPartitionedCall?
IdentityIdentity3classifier_graph_9/StatefulPartitionedCall:output:0+^classifier_graph_9/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????	:::::2X
*classifier_graph_9/StatefulPartitionedCall*classifier_graph_9/StatefulPartitionedCall:O K
'
_output_shapes
:?????????	
 
_user_specified_nameinputs
?!
?
I__inference_sequential_9_layer_call_and_return_conditional_losses_6333642

inputs6
2project_9_matrix_transpose_readvariableop_resource*
&layer_1_matmul_readvariableop_resource+
'layer_1_biasadd_readvariableop_resource)
%output_matmul_readvariableop_resource*
&output_biasadd_readvariableop_resource
identity??layer-1/BiasAdd/ReadVariableOp?layer-1/MatMul/ReadVariableOp?output/BiasAdd/ReadVariableOp?output/MatMul/ReadVariableOp?!project_9/matmul_1/ReadVariableOp?)project_9/matrix_transpose/ReadVariableOp?
)project_9/matrix_transpose/ReadVariableOpReadVariableOp2project_9_matrix_transpose_readvariableop_resource*
_output_shapes

:	*
dtype02+
)project_9/matrix_transpose/ReadVariableOp?
)project_9/matrix_transpose/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2+
)project_9/matrix_transpose/transpose/perm?
$project_9/matrix_transpose/transpose	Transpose1project_9/matrix_transpose/ReadVariableOp:value:02project_9/matrix_transpose/transpose/perm:output:0*
T0*
_output_shapes

:	2&
$project_9/matrix_transpose/transpose?
project_9/matmulMatMulinputs(project_9/matrix_transpose/transpose:y:0*
T0*'
_output_shapes
:?????????2
project_9/matmul?
!project_9/matmul_1/ReadVariableOpReadVariableOp2project_9_matrix_transpose_readvariableop_resource*
_output_shapes

:	*
dtype02#
!project_9/matmul_1/ReadVariableOp?
project_9/matmul_1MatMulproject_9/matmul:product:0)project_9/matmul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	2
project_9/matmul_1}
project_9/subSubinputsproject_9/matmul_1:product:0*
T0*'
_output_shapes
:?????????	2
project_9/sub?
layer-1/MatMul/ReadVariableOpReadVariableOp&layer_1_matmul_readvariableop_resource*
_output_shapes

:	2*
dtype02
layer-1/MatMul/ReadVariableOp?
layer-1/MatMulMatMulproject_9/sub:z:0%layer-1/MatMul/ReadVariableOp:value:0*
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
IdentityIdentityoutput/Softmax:softmax:0^layer-1/BiasAdd/ReadVariableOp^layer-1/MatMul/ReadVariableOp^output/BiasAdd/ReadVariableOp^output/MatMul/ReadVariableOp"^project_9/matmul_1/ReadVariableOp*^project_9/matrix_transpose/ReadVariableOp*
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
!project_9/matmul_1/ReadVariableOp!project_9/matmul_1/ReadVariableOp2V
)project_9/matrix_transpose/ReadVariableOp)project_9/matrix_transpose/ReadVariableOp:O K
'
_output_shapes
:?????????	
 
_user_specified_nameinputs
?!
?
I__inference_sequential_9_layer_call_and_return_conditional_losses_6333724
project_9_input6
2project_9_matrix_transpose_readvariableop_resource*
&layer_1_matmul_readvariableop_resource+
'layer_1_biasadd_readvariableop_resource)
%output_matmul_readvariableop_resource*
&output_biasadd_readvariableop_resource
identity??layer-1/BiasAdd/ReadVariableOp?layer-1/MatMul/ReadVariableOp?output/BiasAdd/ReadVariableOp?output/MatMul/ReadVariableOp?!project_9/matmul_1/ReadVariableOp?)project_9/matrix_transpose/ReadVariableOp?
)project_9/matrix_transpose/ReadVariableOpReadVariableOp2project_9_matrix_transpose_readvariableop_resource*
_output_shapes

:	*
dtype02+
)project_9/matrix_transpose/ReadVariableOp?
)project_9/matrix_transpose/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2+
)project_9/matrix_transpose/transpose/perm?
$project_9/matrix_transpose/transpose	Transpose1project_9/matrix_transpose/ReadVariableOp:value:02project_9/matrix_transpose/transpose/perm:output:0*
T0*
_output_shapes

:	2&
$project_9/matrix_transpose/transpose?
project_9/matmulMatMulproject_9_input(project_9/matrix_transpose/transpose:y:0*
T0*'
_output_shapes
:?????????2
project_9/matmul?
!project_9/matmul_1/ReadVariableOpReadVariableOp2project_9_matrix_transpose_readvariableop_resource*
_output_shapes

:	*
dtype02#
!project_9/matmul_1/ReadVariableOp?
project_9/matmul_1MatMulproject_9/matmul:product:0)project_9/matmul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	2
project_9/matmul_1?
project_9/subSubproject_9_inputproject_9/matmul_1:product:0*
T0*'
_output_shapes
:?????????	2
project_9/sub?
layer-1/MatMul/ReadVariableOpReadVariableOp&layer_1_matmul_readvariableop_resource*
_output_shapes

:	2*
dtype02
layer-1/MatMul/ReadVariableOp?
layer-1/MatMulMatMulproject_9/sub:z:0%layer-1/MatMul/ReadVariableOp:value:0*
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
IdentityIdentityoutput/Softmax:softmax:0^layer-1/BiasAdd/ReadVariableOp^layer-1/MatMul/ReadVariableOp^output/BiasAdd/ReadVariableOp^output/MatMul/ReadVariableOp"^project_9/matmul_1/ReadVariableOp*^project_9/matrix_transpose/ReadVariableOp*
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
!project_9/matmul_1/ReadVariableOp!project_9/matmul_1/ReadVariableOp2V
)project_9/matrix_transpose/ReadVariableOp)project_9/matrix_transpose/ReadVariableOp:X T
'
_output_shapes
:?????????	
)
_user_specified_nameproject_9_input
?
?
 __inference__traced_save_6333832
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
?
?
I__inference_sequential_9_layer_call_and_return_conditional_losses_6333027

inputs
project_9_6333013
layer_1_6333016
layer_1_6333018
output_6333021
output_6333023
identity??layer-1/StatefulPartitionedCall?output/StatefulPartitionedCall?!project_9/StatefulPartitionedCall?
!project_9/StatefulPartitionedCallStatefulPartitionedCallinputsproject_9_6333013*
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
F__inference_project_9_layer_call_and_return_conditional_losses_63329202#
!project_9/StatefulPartitionedCall?
layer-1/StatefulPartitionedCallStatefulPartitionedCall*project_9/StatefulPartitionedCall:output:0layer_1_6333016layer_1_6333018*
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
D__inference_layer-1_layer_call_and_return_conditional_losses_63329462!
layer-1/StatefulPartitionedCall?
output/StatefulPartitionedCallStatefulPartitionedCall(layer-1/StatefulPartitionedCall:output:0output_6333021output_6333023*
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
C__inference_output_layer_call_and_return_conditional_losses_63329732 
output/StatefulPartitionedCall?
IdentityIdentity'output/StatefulPartitionedCall:output:0 ^layer-1/StatefulPartitionedCall^output/StatefulPartitionedCall"^project_9/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????	:::::2B
layer-1/StatefulPartitionedCalllayer-1/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall2F
!project_9/StatefulPartitionedCall!project_9/StatefulPartitionedCall:O K
'
_output_shapes
:?????????	
 
_user_specified_nameinputs
?
?
)__inference_model_9_layer_call_fn_6333297
input_10
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_10unknown	unknown_0	unknown_1	unknown_2	unknown_3*
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
D__inference_model_9_layer_call_and_return_conditional_losses_63332842
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????	:::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:?????????	
"
_user_specified_name
input_10
?
?
#__inference__traced_restore_6333857
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
?

?
D__inference_model_9_layer_call_and_return_conditional_losses_6333266
input_10
classifier_graph_9_6333254
classifier_graph_9_6333256
classifier_graph_9_6333258
classifier_graph_9_6333260
classifier_graph_9_6333262
identity??*classifier_graph_9/StatefulPartitionedCall?
*classifier_graph_9/StatefulPartitionedCallStatefulPartitionedCallinput_10classifier_graph_9_6333254classifier_graph_9_6333256classifier_graph_9_6333258classifier_graph_9_6333260classifier_graph_9_6333262*
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
O__inference_classifier_graph_9_layer_call_and_return_conditional_losses_63331512,
*classifier_graph_9/StatefulPartitionedCall?
IdentityIdentity3classifier_graph_9/StatefulPartitionedCall:output:0+^classifier_graph_9/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????	:::::2X
*classifier_graph_9/StatefulPartitionedCall*classifier_graph_9/StatefulPartitionedCall:Q M
'
_output_shapes
:?????????	
"
_user_specified_name
input_10
?	
?
D__inference_layer-1_layer_call_and_return_conditional_losses_6332946

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
)__inference_model_9_layer_call_fn_6333327
input_10
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_10unknown	unknown_0	unknown_1	unknown_2	unknown_3*
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
D__inference_model_9_layer_call_and_return_conditional_losses_63333142
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????	:::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:?????????	
"
_user_specified_name
input_10
?)
?
O__inference_classifier_graph_9_layer_call_and_return_conditional_losses_6333452
input_1C
?sequential_9_project_9_matrix_transpose_readvariableop_resource7
3sequential_9_layer_1_matmul_readvariableop_resource8
4sequential_9_layer_1_biasadd_readvariableop_resource6
2sequential_9_output_matmul_readvariableop_resource7
3sequential_9_output_biasadd_readvariableop_resource
identity??+sequential_9/layer-1/BiasAdd/ReadVariableOp?*sequential_9/layer-1/MatMul/ReadVariableOp?*sequential_9/output/BiasAdd/ReadVariableOp?)sequential_9/output/MatMul/ReadVariableOp?.sequential_9/project_9/matmul_1/ReadVariableOp?6sequential_9/project_9/matrix_transpose/ReadVariableOp?
6sequential_9/project_9/matrix_transpose/ReadVariableOpReadVariableOp?sequential_9_project_9_matrix_transpose_readvariableop_resource*
_output_shapes

:	*
dtype028
6sequential_9/project_9/matrix_transpose/ReadVariableOp?
6sequential_9/project_9/matrix_transpose/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       28
6sequential_9/project_9/matrix_transpose/transpose/perm?
1sequential_9/project_9/matrix_transpose/transpose	Transpose>sequential_9/project_9/matrix_transpose/ReadVariableOp:value:0?sequential_9/project_9/matrix_transpose/transpose/perm:output:0*
T0*
_output_shapes

:	23
1sequential_9/project_9/matrix_transpose/transpose?
sequential_9/project_9/matmulMatMulinput_15sequential_9/project_9/matrix_transpose/transpose:y:0*
T0*'
_output_shapes
:?????????2
sequential_9/project_9/matmul?
.sequential_9/project_9/matmul_1/ReadVariableOpReadVariableOp?sequential_9_project_9_matrix_transpose_readvariableop_resource*
_output_shapes

:	*
dtype020
.sequential_9/project_9/matmul_1/ReadVariableOp?
sequential_9/project_9/matmul_1MatMul'sequential_9/project_9/matmul:product:06sequential_9/project_9/matmul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	2!
sequential_9/project_9/matmul_1?
sequential_9/project_9/subSubinput_1)sequential_9/project_9/matmul_1:product:0*
T0*'
_output_shapes
:?????????	2
sequential_9/project_9/sub?
*sequential_9/layer-1/MatMul/ReadVariableOpReadVariableOp3sequential_9_layer_1_matmul_readvariableop_resource*
_output_shapes

:	2*
dtype02,
*sequential_9/layer-1/MatMul/ReadVariableOp?
sequential_9/layer-1/MatMulMatMulsequential_9/project_9/sub:z:02sequential_9/layer-1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
sequential_9/layer-1/MatMul?
+sequential_9/layer-1/BiasAdd/ReadVariableOpReadVariableOp4sequential_9_layer_1_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02-
+sequential_9/layer-1/BiasAdd/ReadVariableOp?
sequential_9/layer-1/BiasAddBiasAdd%sequential_9/layer-1/MatMul:product:03sequential_9/layer-1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
sequential_9/layer-1/BiasAdd?
sequential_9/layer-1/ReluRelu%sequential_9/layer-1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????22
sequential_9/layer-1/Relu?
)sequential_9/output/MatMul/ReadVariableOpReadVariableOp2sequential_9_output_matmul_readvariableop_resource*
_output_shapes

:2*
dtype02+
)sequential_9/output/MatMul/ReadVariableOp?
sequential_9/output/MatMulMatMul'sequential_9/layer-1/Relu:activations:01sequential_9/output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential_9/output/MatMul?
*sequential_9/output/BiasAdd/ReadVariableOpReadVariableOp3sequential_9_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02,
*sequential_9/output/BiasAdd/ReadVariableOp?
sequential_9/output/BiasAddBiasAdd$sequential_9/output/MatMul:product:02sequential_9/output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential_9/output/BiasAdd?
sequential_9/output/SoftmaxSoftmax$sequential_9/output/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
sequential_9/output/Softmax?
IdentityIdentity%sequential_9/output/Softmax:softmax:0,^sequential_9/layer-1/BiasAdd/ReadVariableOp+^sequential_9/layer-1/MatMul/ReadVariableOp+^sequential_9/output/BiasAdd/ReadVariableOp*^sequential_9/output/MatMul/ReadVariableOp/^sequential_9/project_9/matmul_1/ReadVariableOp7^sequential_9/project_9/matrix_transpose/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????	:::::2Z
+sequential_9/layer-1/BiasAdd/ReadVariableOp+sequential_9/layer-1/BiasAdd/ReadVariableOp2X
*sequential_9/layer-1/MatMul/ReadVariableOp*sequential_9/layer-1/MatMul/ReadVariableOp2X
*sequential_9/output/BiasAdd/ReadVariableOp*sequential_9/output/BiasAdd/ReadVariableOp2V
)sequential_9/output/MatMul/ReadVariableOp)sequential_9/output/MatMul/ReadVariableOp2`
.sequential_9/project_9/matmul_1/ReadVariableOp.sequential_9/project_9/matmul_1/ReadVariableOp2p
6sequential_9/project_9/matrix_transpose/ReadVariableOp6sequential_9/project_9/matrix_transpose/ReadVariableOp:P L
'
_output_shapes
:?????????	
!
_user_specified_name	input_1
?
?
)__inference_model_9_layer_call_fn_6333426

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
D__inference_model_9_layer_call_and_return_conditional_losses_63333142
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
O__inference_classifier_graph_9_layer_call_and_return_conditional_losses_6333151
x
sequential_9_6333139
sequential_9_6333141
sequential_9_6333143
sequential_9_6333145
sequential_9_6333147
identity??$sequential_9/StatefulPartitionedCall?
$sequential_9/StatefulPartitionedCallStatefulPartitionedCallxsequential_9_6333139sequential_9_6333141sequential_9_6333143sequential_9_6333145sequential_9_6333147*
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
I__inference_sequential_9_layer_call_and_return_conditional_losses_63330592&
$sequential_9/StatefulPartitionedCall?
IdentityIdentity-sequential_9/StatefulPartitionedCall:output:0%^sequential_9/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????	:::::2L
$sequential_9/StatefulPartitionedCall$sequential_9/StatefulPartitionedCall:J F
'
_output_shapes
:?????????	

_user_specified_namex
?
?
.__inference_sequential_9_layer_call_fn_6333672

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
I__inference_sequential_9_layer_call_and_return_conditional_losses_63330592
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
.__inference_sequential_9_layer_call_fn_6333657

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
I__inference_sequential_9_layer_call_and_return_conditional_losses_63330272
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
.__inference_sequential_9_layer_call_fn_6333739
project_9_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallproject_9_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3*
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
I__inference_sequential_9_layer_call_and_return_conditional_losses_63330272
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
_user_specified_nameproject_9_input
?
?
F__inference_project_9_layer_call_and_return_conditional_losses_6332920
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
?)
?
O__inference_classifier_graph_9_layer_call_and_return_conditional_losses_6333534
xC
?sequential_9_project_9_matrix_transpose_readvariableop_resource7
3sequential_9_layer_1_matmul_readvariableop_resource8
4sequential_9_layer_1_biasadd_readvariableop_resource6
2sequential_9_output_matmul_readvariableop_resource7
3sequential_9_output_biasadd_readvariableop_resource
identity??+sequential_9/layer-1/BiasAdd/ReadVariableOp?*sequential_9/layer-1/MatMul/ReadVariableOp?*sequential_9/output/BiasAdd/ReadVariableOp?)sequential_9/output/MatMul/ReadVariableOp?.sequential_9/project_9/matmul_1/ReadVariableOp?6sequential_9/project_9/matrix_transpose/ReadVariableOp?
6sequential_9/project_9/matrix_transpose/ReadVariableOpReadVariableOp?sequential_9_project_9_matrix_transpose_readvariableop_resource*
_output_shapes

:	*
dtype028
6sequential_9/project_9/matrix_transpose/ReadVariableOp?
6sequential_9/project_9/matrix_transpose/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       28
6sequential_9/project_9/matrix_transpose/transpose/perm?
1sequential_9/project_9/matrix_transpose/transpose	Transpose>sequential_9/project_9/matrix_transpose/ReadVariableOp:value:0?sequential_9/project_9/matrix_transpose/transpose/perm:output:0*
T0*
_output_shapes

:	23
1sequential_9/project_9/matrix_transpose/transpose?
sequential_9/project_9/matmulMatMulx5sequential_9/project_9/matrix_transpose/transpose:y:0*
T0*'
_output_shapes
:?????????2
sequential_9/project_9/matmul?
.sequential_9/project_9/matmul_1/ReadVariableOpReadVariableOp?sequential_9_project_9_matrix_transpose_readvariableop_resource*
_output_shapes

:	*
dtype020
.sequential_9/project_9/matmul_1/ReadVariableOp?
sequential_9/project_9/matmul_1MatMul'sequential_9/project_9/matmul:product:06sequential_9/project_9/matmul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	2!
sequential_9/project_9/matmul_1?
sequential_9/project_9/subSubx)sequential_9/project_9/matmul_1:product:0*
T0*'
_output_shapes
:?????????	2
sequential_9/project_9/sub?
*sequential_9/layer-1/MatMul/ReadVariableOpReadVariableOp3sequential_9_layer_1_matmul_readvariableop_resource*
_output_shapes

:	2*
dtype02,
*sequential_9/layer-1/MatMul/ReadVariableOp?
sequential_9/layer-1/MatMulMatMulsequential_9/project_9/sub:z:02sequential_9/layer-1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
sequential_9/layer-1/MatMul?
+sequential_9/layer-1/BiasAdd/ReadVariableOpReadVariableOp4sequential_9_layer_1_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02-
+sequential_9/layer-1/BiasAdd/ReadVariableOp?
sequential_9/layer-1/BiasAddBiasAdd%sequential_9/layer-1/MatMul:product:03sequential_9/layer-1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
sequential_9/layer-1/BiasAdd?
sequential_9/layer-1/ReluRelu%sequential_9/layer-1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????22
sequential_9/layer-1/Relu?
)sequential_9/output/MatMul/ReadVariableOpReadVariableOp2sequential_9_output_matmul_readvariableop_resource*
_output_shapes

:2*
dtype02+
)sequential_9/output/MatMul/ReadVariableOp?
sequential_9/output/MatMulMatMul'sequential_9/layer-1/Relu:activations:01sequential_9/output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential_9/output/MatMul?
*sequential_9/output/BiasAdd/ReadVariableOpReadVariableOp3sequential_9_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02,
*sequential_9/output/BiasAdd/ReadVariableOp?
sequential_9/output/BiasAddBiasAdd$sequential_9/output/MatMul:product:02sequential_9/output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential_9/output/BiasAdd?
sequential_9/output/SoftmaxSoftmax$sequential_9/output/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
sequential_9/output/Softmax?
IdentityIdentity%sequential_9/output/Softmax:softmax:0,^sequential_9/layer-1/BiasAdd/ReadVariableOp+^sequential_9/layer-1/MatMul/ReadVariableOp+^sequential_9/output/BiasAdd/ReadVariableOp*^sequential_9/output/MatMul/ReadVariableOp/^sequential_9/project_9/matmul_1/ReadVariableOp7^sequential_9/project_9/matrix_transpose/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????	:::::2Z
+sequential_9/layer-1/BiasAdd/ReadVariableOp+sequential_9/layer-1/BiasAdd/ReadVariableOp2X
*sequential_9/layer-1/MatMul/ReadVariableOp*sequential_9/layer-1/MatMul/ReadVariableOp2X
*sequential_9/output/BiasAdd/ReadVariableOp*sequential_9/output/BiasAdd/ReadVariableOp2V
)sequential_9/output/MatMul/ReadVariableOp)sequential_9/output/MatMul/ReadVariableOp2`
.sequential_9/project_9/matmul_1/ReadVariableOp.sequential_9/project_9/matmul_1/ReadVariableOp2p
6sequential_9/project_9/matrix_transpose/ReadVariableOp6sequential_9/project_9/matrix_transpose/ReadVariableOp:J F
'
_output_shapes
:?????????	

_user_specified_namex
?	
?
C__inference_output_layer_call_and_return_conditional_losses_6332973

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
?!
?
I__inference_sequential_9_layer_call_and_return_conditional_losses_6333698
project_9_input6
2project_9_matrix_transpose_readvariableop_resource*
&layer_1_matmul_readvariableop_resource+
'layer_1_biasadd_readvariableop_resource)
%output_matmul_readvariableop_resource*
&output_biasadd_readvariableop_resource
identity??layer-1/BiasAdd/ReadVariableOp?layer-1/MatMul/ReadVariableOp?output/BiasAdd/ReadVariableOp?output/MatMul/ReadVariableOp?!project_9/matmul_1/ReadVariableOp?)project_9/matrix_transpose/ReadVariableOp?
)project_9/matrix_transpose/ReadVariableOpReadVariableOp2project_9_matrix_transpose_readvariableop_resource*
_output_shapes

:	*
dtype02+
)project_9/matrix_transpose/ReadVariableOp?
)project_9/matrix_transpose/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2+
)project_9/matrix_transpose/transpose/perm?
$project_9/matrix_transpose/transpose	Transpose1project_9/matrix_transpose/ReadVariableOp:value:02project_9/matrix_transpose/transpose/perm:output:0*
T0*
_output_shapes

:	2&
$project_9/matrix_transpose/transpose?
project_9/matmulMatMulproject_9_input(project_9/matrix_transpose/transpose:y:0*
T0*'
_output_shapes
:?????????2
project_9/matmul?
!project_9/matmul_1/ReadVariableOpReadVariableOp2project_9_matrix_transpose_readvariableop_resource*
_output_shapes

:	*
dtype02#
!project_9/matmul_1/ReadVariableOp?
project_9/matmul_1MatMulproject_9/matmul:product:0)project_9/matmul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	2
project_9/matmul_1?
project_9/subSubproject_9_inputproject_9/matmul_1:product:0*
T0*'
_output_shapes
:?????????	2
project_9/sub?
layer-1/MatMul/ReadVariableOpReadVariableOp&layer_1_matmul_readvariableop_resource*
_output_shapes

:	2*
dtype02
layer-1/MatMul/ReadVariableOp?
layer-1/MatMulMatMulproject_9/sub:z:0%layer-1/MatMul/ReadVariableOp:value:0*
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
IdentityIdentityoutput/Softmax:softmax:0^layer-1/BiasAdd/ReadVariableOp^layer-1/MatMul/ReadVariableOp^output/BiasAdd/ReadVariableOp^output/MatMul/ReadVariableOp"^project_9/matmul_1/ReadVariableOp*^project_9/matrix_transpose/ReadVariableOp*
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
!project_9/matmul_1/ReadVariableOp!project_9/matmul_1/ReadVariableOp2V
)project_9/matrix_transpose/ReadVariableOp)project_9/matrix_transpose/ReadVariableOp:X T
'
_output_shapes
:?????????	
)
_user_specified_nameproject_9_input
?
?
4__inference_classifier_graph_9_layer_call_fn_6333575
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
O__inference_classifier_graph_9_layer_call_and_return_conditional_losses_63331512
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
4__inference_classifier_graph_9_layer_call_fn_6333508
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
O__inference_classifier_graph_9_layer_call_and_return_conditional_losses_63331512
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
?)
?
O__inference_classifier_graph_9_layer_call_and_return_conditional_losses_6333560
xC
?sequential_9_project_9_matrix_transpose_readvariableop_resource7
3sequential_9_layer_1_matmul_readvariableop_resource8
4sequential_9_layer_1_biasadd_readvariableop_resource6
2sequential_9_output_matmul_readvariableop_resource7
3sequential_9_output_biasadd_readvariableop_resource
identity??+sequential_9/layer-1/BiasAdd/ReadVariableOp?*sequential_9/layer-1/MatMul/ReadVariableOp?*sequential_9/output/BiasAdd/ReadVariableOp?)sequential_9/output/MatMul/ReadVariableOp?.sequential_9/project_9/matmul_1/ReadVariableOp?6sequential_9/project_9/matrix_transpose/ReadVariableOp?
6sequential_9/project_9/matrix_transpose/ReadVariableOpReadVariableOp?sequential_9_project_9_matrix_transpose_readvariableop_resource*
_output_shapes

:	*
dtype028
6sequential_9/project_9/matrix_transpose/ReadVariableOp?
6sequential_9/project_9/matrix_transpose/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       28
6sequential_9/project_9/matrix_transpose/transpose/perm?
1sequential_9/project_9/matrix_transpose/transpose	Transpose>sequential_9/project_9/matrix_transpose/ReadVariableOp:value:0?sequential_9/project_9/matrix_transpose/transpose/perm:output:0*
T0*
_output_shapes

:	23
1sequential_9/project_9/matrix_transpose/transpose?
sequential_9/project_9/matmulMatMulx5sequential_9/project_9/matrix_transpose/transpose:y:0*
T0*'
_output_shapes
:?????????2
sequential_9/project_9/matmul?
.sequential_9/project_9/matmul_1/ReadVariableOpReadVariableOp?sequential_9_project_9_matrix_transpose_readvariableop_resource*
_output_shapes

:	*
dtype020
.sequential_9/project_9/matmul_1/ReadVariableOp?
sequential_9/project_9/matmul_1MatMul'sequential_9/project_9/matmul:product:06sequential_9/project_9/matmul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	2!
sequential_9/project_9/matmul_1?
sequential_9/project_9/subSubx)sequential_9/project_9/matmul_1:product:0*
T0*'
_output_shapes
:?????????	2
sequential_9/project_9/sub?
*sequential_9/layer-1/MatMul/ReadVariableOpReadVariableOp3sequential_9_layer_1_matmul_readvariableop_resource*
_output_shapes

:	2*
dtype02,
*sequential_9/layer-1/MatMul/ReadVariableOp?
sequential_9/layer-1/MatMulMatMulsequential_9/project_9/sub:z:02sequential_9/layer-1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
sequential_9/layer-1/MatMul?
+sequential_9/layer-1/BiasAdd/ReadVariableOpReadVariableOp4sequential_9_layer_1_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02-
+sequential_9/layer-1/BiasAdd/ReadVariableOp?
sequential_9/layer-1/BiasAddBiasAdd%sequential_9/layer-1/MatMul:product:03sequential_9/layer-1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
sequential_9/layer-1/BiasAdd?
sequential_9/layer-1/ReluRelu%sequential_9/layer-1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????22
sequential_9/layer-1/Relu?
)sequential_9/output/MatMul/ReadVariableOpReadVariableOp2sequential_9_output_matmul_readvariableop_resource*
_output_shapes

:2*
dtype02+
)sequential_9/output/MatMul/ReadVariableOp?
sequential_9/output/MatMulMatMul'sequential_9/layer-1/Relu:activations:01sequential_9/output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential_9/output/MatMul?
*sequential_9/output/BiasAdd/ReadVariableOpReadVariableOp3sequential_9_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02,
*sequential_9/output/BiasAdd/ReadVariableOp?
sequential_9/output/BiasAddBiasAdd$sequential_9/output/MatMul:product:02sequential_9/output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential_9/output/BiasAdd?
sequential_9/output/SoftmaxSoftmax$sequential_9/output/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
sequential_9/output/Softmax?
IdentityIdentity%sequential_9/output/Softmax:softmax:0,^sequential_9/layer-1/BiasAdd/ReadVariableOp+^sequential_9/layer-1/MatMul/ReadVariableOp+^sequential_9/output/BiasAdd/ReadVariableOp*^sequential_9/output/MatMul/ReadVariableOp/^sequential_9/project_9/matmul_1/ReadVariableOp7^sequential_9/project_9/matrix_transpose/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????	:::::2Z
+sequential_9/layer-1/BiasAdd/ReadVariableOp+sequential_9/layer-1/BiasAdd/ReadVariableOp2X
*sequential_9/layer-1/MatMul/ReadVariableOp*sequential_9/layer-1/MatMul/ReadVariableOp2X
*sequential_9/output/BiasAdd/ReadVariableOp*sequential_9/output/BiasAdd/ReadVariableOp2V
)sequential_9/output/MatMul/ReadVariableOp)sequential_9/output/MatMul/ReadVariableOp2`
.sequential_9/project_9/matmul_1/ReadVariableOp.sequential_9/project_9/matmul_1/ReadVariableOp2p
6sequential_9/project_9/matrix_transpose/ReadVariableOp6sequential_9/project_9/matrix_transpose/ReadVariableOp:J F
'
_output_shapes
:?????????	

_user_specified_namex
?6
?
D__inference_model_9_layer_call_and_return_conditional_losses_6333370

inputsV
Rclassifier_graph_9_sequential_9_project_9_matrix_transpose_readvariableop_resourceJ
Fclassifier_graph_9_sequential_9_layer_1_matmul_readvariableop_resourceK
Gclassifier_graph_9_sequential_9_layer_1_biasadd_readvariableop_resourceI
Eclassifier_graph_9_sequential_9_output_matmul_readvariableop_resourceJ
Fclassifier_graph_9_sequential_9_output_biasadd_readvariableop_resource
identity??>classifier_graph_9/sequential_9/layer-1/BiasAdd/ReadVariableOp?=classifier_graph_9/sequential_9/layer-1/MatMul/ReadVariableOp?=classifier_graph_9/sequential_9/output/BiasAdd/ReadVariableOp?<classifier_graph_9/sequential_9/output/MatMul/ReadVariableOp?Aclassifier_graph_9/sequential_9/project_9/matmul_1/ReadVariableOp?Iclassifier_graph_9/sequential_9/project_9/matrix_transpose/ReadVariableOp?
Iclassifier_graph_9/sequential_9/project_9/matrix_transpose/ReadVariableOpReadVariableOpRclassifier_graph_9_sequential_9_project_9_matrix_transpose_readvariableop_resource*
_output_shapes

:	*
dtype02K
Iclassifier_graph_9/sequential_9/project_9/matrix_transpose/ReadVariableOp?
Iclassifier_graph_9/sequential_9/project_9/matrix_transpose/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2K
Iclassifier_graph_9/sequential_9/project_9/matrix_transpose/transpose/perm?
Dclassifier_graph_9/sequential_9/project_9/matrix_transpose/transpose	TransposeQclassifier_graph_9/sequential_9/project_9/matrix_transpose/ReadVariableOp:value:0Rclassifier_graph_9/sequential_9/project_9/matrix_transpose/transpose/perm:output:0*
T0*
_output_shapes

:	2F
Dclassifier_graph_9/sequential_9/project_9/matrix_transpose/transpose?
0classifier_graph_9/sequential_9/project_9/matmulMatMulinputsHclassifier_graph_9/sequential_9/project_9/matrix_transpose/transpose:y:0*
T0*'
_output_shapes
:?????????22
0classifier_graph_9/sequential_9/project_9/matmul?
Aclassifier_graph_9/sequential_9/project_9/matmul_1/ReadVariableOpReadVariableOpRclassifier_graph_9_sequential_9_project_9_matrix_transpose_readvariableop_resource*
_output_shapes

:	*
dtype02C
Aclassifier_graph_9/sequential_9/project_9/matmul_1/ReadVariableOp?
2classifier_graph_9/sequential_9/project_9/matmul_1MatMul:classifier_graph_9/sequential_9/project_9/matmul:product:0Iclassifier_graph_9/sequential_9/project_9/matmul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	24
2classifier_graph_9/sequential_9/project_9/matmul_1?
-classifier_graph_9/sequential_9/project_9/subSubinputs<classifier_graph_9/sequential_9/project_9/matmul_1:product:0*
T0*'
_output_shapes
:?????????	2/
-classifier_graph_9/sequential_9/project_9/sub?
=classifier_graph_9/sequential_9/layer-1/MatMul/ReadVariableOpReadVariableOpFclassifier_graph_9_sequential_9_layer_1_matmul_readvariableop_resource*
_output_shapes

:	2*
dtype02?
=classifier_graph_9/sequential_9/layer-1/MatMul/ReadVariableOp?
.classifier_graph_9/sequential_9/layer-1/MatMulMatMul1classifier_graph_9/sequential_9/project_9/sub:z:0Eclassifier_graph_9/sequential_9/layer-1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????220
.classifier_graph_9/sequential_9/layer-1/MatMul?
>classifier_graph_9/sequential_9/layer-1/BiasAdd/ReadVariableOpReadVariableOpGclassifier_graph_9_sequential_9_layer_1_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02@
>classifier_graph_9/sequential_9/layer-1/BiasAdd/ReadVariableOp?
/classifier_graph_9/sequential_9/layer-1/BiasAddBiasAdd8classifier_graph_9/sequential_9/layer-1/MatMul:product:0Fclassifier_graph_9/sequential_9/layer-1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????221
/classifier_graph_9/sequential_9/layer-1/BiasAdd?
,classifier_graph_9/sequential_9/layer-1/ReluRelu8classifier_graph_9/sequential_9/layer-1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????22.
,classifier_graph_9/sequential_9/layer-1/Relu?
<classifier_graph_9/sequential_9/output/MatMul/ReadVariableOpReadVariableOpEclassifier_graph_9_sequential_9_output_matmul_readvariableop_resource*
_output_shapes

:2*
dtype02>
<classifier_graph_9/sequential_9/output/MatMul/ReadVariableOp?
-classifier_graph_9/sequential_9/output/MatMulMatMul:classifier_graph_9/sequential_9/layer-1/Relu:activations:0Dclassifier_graph_9/sequential_9/output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2/
-classifier_graph_9/sequential_9/output/MatMul?
=classifier_graph_9/sequential_9/output/BiasAdd/ReadVariableOpReadVariableOpFclassifier_graph_9_sequential_9_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02?
=classifier_graph_9/sequential_9/output/BiasAdd/ReadVariableOp?
.classifier_graph_9/sequential_9/output/BiasAddBiasAdd7classifier_graph_9/sequential_9/output/MatMul:product:0Eclassifier_graph_9/sequential_9/output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????20
.classifier_graph_9/sequential_9/output/BiasAdd?
.classifier_graph_9/sequential_9/output/SoftmaxSoftmax7classifier_graph_9/sequential_9/output/BiasAdd:output:0*
T0*'
_output_shapes
:?????????20
.classifier_graph_9/sequential_9/output/Softmax?
IdentityIdentity8classifier_graph_9/sequential_9/output/Softmax:softmax:0?^classifier_graph_9/sequential_9/layer-1/BiasAdd/ReadVariableOp>^classifier_graph_9/sequential_9/layer-1/MatMul/ReadVariableOp>^classifier_graph_9/sequential_9/output/BiasAdd/ReadVariableOp=^classifier_graph_9/sequential_9/output/MatMul/ReadVariableOpB^classifier_graph_9/sequential_9/project_9/matmul_1/ReadVariableOpJ^classifier_graph_9/sequential_9/project_9/matrix_transpose/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????	:::::2?
>classifier_graph_9/sequential_9/layer-1/BiasAdd/ReadVariableOp>classifier_graph_9/sequential_9/layer-1/BiasAdd/ReadVariableOp2~
=classifier_graph_9/sequential_9/layer-1/MatMul/ReadVariableOp=classifier_graph_9/sequential_9/layer-1/MatMul/ReadVariableOp2~
=classifier_graph_9/sequential_9/output/BiasAdd/ReadVariableOp=classifier_graph_9/sequential_9/output/BiasAdd/ReadVariableOp2|
<classifier_graph_9/sequential_9/output/MatMul/ReadVariableOp<classifier_graph_9/sequential_9/output/MatMul/ReadVariableOp2?
Aclassifier_graph_9/sequential_9/project_9/matmul_1/ReadVariableOpAclassifier_graph_9/sequential_9/project_9/matmul_1/ReadVariableOp2?
Iclassifier_graph_9/sequential_9/project_9/matrix_transpose/ReadVariableOpIclassifier_graph_9/sequential_9/project_9/matrix_transpose/ReadVariableOp:O K
'
_output_shapes
:?????????	
 
_user_specified_nameinputs
?
?
4__inference_classifier_graph_9_layer_call_fn_6333493
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
O__inference_classifier_graph_9_layer_call_and_return_conditional_losses_63331512
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
_user_specified_name	input_1"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
=
input_101
serving_default_input_10:0?????????	F
classifier_graph_90
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
_tf_keras_network?{"class_name": "Functional", "name": "model_9", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "model_9", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 9]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_10"}, "name": "input_10", "inbound_nodes": []}, {"class_name": "ClassifierGraph", "config": {"layer was saved without config": true}, "name": "classifier_graph_9", "inbound_nodes": [[["input_10", 0, 0, {}]]]}], "input_layers": [["input_10", 0, 0]], "output_layers": [["classifier_graph_9", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 9]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 9]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional"}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input_10", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 9]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 9]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_10"}}
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
_tf_keras_model?{"class_name": "ClassifierGraph", "name": "classifier_graph_9", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "is_graph_network": false, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "ClassifierGraph"}}
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
_tf_keras_sequential?{"class_name": "Sequential", "name": "sequential_9", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_9", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 9]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "project_9_input"}}, {"class_name": "Project", "config": {"layer was saved without config": true}}, {"class_name": "Dense", "config": {"name": "layer-1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 9]}, "dtype": "float32", "units": 50, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "output", "trainable": true, "dtype": "float32", "units": 2, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 9]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 9]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential"}}
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
_tf_keras_layer?{"class_name": "Project", "name": "project_9", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 9]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}}
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
"__inference__wrapped_model_6332907?
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
annotations? *'?$
"?
input_10?????????	
?2?
D__inference_model_9_layer_call_and_return_conditional_losses_6333370
D__inference_model_9_layer_call_and_return_conditional_losses_6333396
D__inference_model_9_layer_call_and_return_conditional_losses_6333266
D__inference_model_9_layer_call_and_return_conditional_losses_6333251?
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
)__inference_model_9_layer_call_fn_6333327
)__inference_model_9_layer_call_fn_6333411
)__inference_model_9_layer_call_fn_6333297
)__inference_model_9_layer_call_fn_6333426?
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
O__inference_classifier_graph_9_layer_call_and_return_conditional_losses_6333478
O__inference_classifier_graph_9_layer_call_and_return_conditional_losses_6333534
O__inference_classifier_graph_9_layer_call_and_return_conditional_losses_6333560
O__inference_classifier_graph_9_layer_call_and_return_conditional_losses_6333452?
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
4__inference_classifier_graph_9_layer_call_fn_6333493
4__inference_classifier_graph_9_layer_call_fn_6333590
4__inference_classifier_graph_9_layer_call_fn_6333508
4__inference_classifier_graph_9_layer_call_fn_6333575?
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
%__inference_signature_wrapper_6333344input_10"?
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
I__inference_sequential_9_layer_call_and_return_conditional_losses_6333642
I__inference_sequential_9_layer_call_and_return_conditional_losses_6333724
I__inference_sequential_9_layer_call_and_return_conditional_losses_6333616
I__inference_sequential_9_layer_call_and_return_conditional_losses_6333698?
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
.__inference_sequential_9_layer_call_fn_6333754
.__inference_sequential_9_layer_call_fn_6333672
.__inference_sequential_9_layer_call_fn_6333657
.__inference_sequential_9_layer_call_fn_6333739?
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
F__inference_project_9_layer_call_and_return_conditional_losses_6332920?
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
+__inference_project_9_layer_call_fn_6332928?
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
D__inference_layer-1_layer_call_and_return_conditional_losses_6333765?
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
)__inference_layer-1_layer_call_fn_6333774?
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
C__inference_output_layer_call_and_return_conditional_losses_6333785?
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
(__inference_output_layer_call_fn_6333794?
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
"__inference__wrapped_model_6332907?1?.
'?$
"?
input_10?????????	
? "G?D
B
classifier_graph_9,?)
classifier_graph_9??????????
O__inference_classifier_graph_9_layer_call_and_return_conditional_losses_6333452h8?5
.?+
!?
input_1?????????	
p 
p
? "%?"
?
0?????????
? ?
O__inference_classifier_graph_9_layer_call_and_return_conditional_losses_6333478h8?5
.?+
!?
input_1?????????	
p 
p 
? "%?"
?
0?????????
? ?
O__inference_classifier_graph_9_layer_call_and_return_conditional_losses_6333534b2?/
(?%
?
x?????????	
p 
p
? "%?"
?
0?????????
? ?
O__inference_classifier_graph_9_layer_call_and_return_conditional_losses_6333560b2?/
(?%
?
x?????????	
p 
p 
? "%?"
?
0?????????
? ?
4__inference_classifier_graph_9_layer_call_fn_6333493[8?5
.?+
!?
input_1?????????	
p 
p
? "???????????
4__inference_classifier_graph_9_layer_call_fn_6333508[8?5
.?+
!?
input_1?????????	
p 
p 
? "???????????
4__inference_classifier_graph_9_layer_call_fn_6333575U2?/
(?%
?
x?????????	
p 
p
? "???????????
4__inference_classifier_graph_9_layer_call_fn_6333590U2?/
(?%
?
x?????????	
p 
p 
? "???????????
D__inference_layer-1_layer_call_and_return_conditional_losses_6333765\/?,
%?"
 ?
inputs?????????	
? "%?"
?
0?????????2
? |
)__inference_layer-1_layer_call_fn_6333774O/?,
%?"
 ?
inputs?????????	
? "??????????2?
D__inference_model_9_layer_call_and_return_conditional_losses_6333251i9?6
/?,
"?
input_10?????????	
p

 
? "%?"
?
0?????????
? ?
D__inference_model_9_layer_call_and_return_conditional_losses_6333266i9?6
/?,
"?
input_10?????????	
p 

 
? "%?"
?
0?????????
? ?
D__inference_model_9_layer_call_and_return_conditional_losses_6333370g7?4
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
D__inference_model_9_layer_call_and_return_conditional_losses_6333396g7?4
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
)__inference_model_9_layer_call_fn_6333297\9?6
/?,
"?
input_10?????????	
p

 
? "???????????
)__inference_model_9_layer_call_fn_6333327\9?6
/?,
"?
input_10?????????	
p 

 
? "???????????
)__inference_model_9_layer_call_fn_6333411Z7?4
-?*
 ?
inputs?????????	
p

 
? "???????????
)__inference_model_9_layer_call_fn_6333426Z7?4
-?*
 ?
inputs?????????	
p 

 
? "???????????
C__inference_output_layer_call_and_return_conditional_losses_6333785\/?,
%?"
 ?
inputs?????????2
? "%?"
?
0?????????
? {
(__inference_output_layer_call_fn_6333794O/?,
%?"
 ?
inputs?????????2
? "???????????
F__inference_project_9_layer_call_and_return_conditional_losses_6332920V*?'
 ?
?
x?????????	
? "%?"
?
0?????????	
? x
+__inference_project_9_layer_call_fn_6332928I*?'
 ?
?
x?????????	
? "??????????	?
I__inference_sequential_9_layer_call_and_return_conditional_losses_6333616g7?4
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
I__inference_sequential_9_layer_call_and_return_conditional_losses_6333642g7?4
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
I__inference_sequential_9_layer_call_and_return_conditional_losses_6333698p@?=
6?3
)?&
project_9_input?????????	
p

 
? "%?"
?
0?????????
? ?
I__inference_sequential_9_layer_call_and_return_conditional_losses_6333724p@?=
6?3
)?&
project_9_input?????????	
p 

 
? "%?"
?
0?????????
? ?
.__inference_sequential_9_layer_call_fn_6333657Z7?4
-?*
 ?
inputs?????????	
p

 
? "???????????
.__inference_sequential_9_layer_call_fn_6333672Z7?4
-?*
 ?
inputs?????????	
p 

 
? "???????????
.__inference_sequential_9_layer_call_fn_6333739c@?=
6?3
)?&
project_9_input?????????	
p

 
? "???????????
.__inference_sequential_9_layer_call_fn_6333754c@?=
6?3
)?&
project_9_input?????????	
p 

 
? "???????????
%__inference_signature_wrapper_6333344?=?:
? 
3?0
.
input_10"?
input_10?????????	"G?D
B
classifier_graph_9,?)
classifier_graph_9?????????