Ɉ
??
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
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
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape?"serve*2.1.02v2.1.0-rc2-17-ge5bf8de4108??
?
.classifier_graph_1/sequential_1/layer-1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2*?
shared_name0.classifier_graph_1/sequential_1/layer-1/kernel
?
Bclassifier_graph_1/sequential_1/layer-1/kernel/Read/ReadVariableOpReadVariableOp.classifier_graph_1/sequential_1/layer-1/kernel*
_output_shapes

:2*
dtype0
?
,classifier_graph_1/sequential_1/layer-1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*=
shared_name.,classifier_graph_1/sequential_1/layer-1/bias
?
@classifier_graph_1/sequential_1/layer-1/bias/Read/ReadVariableOpReadVariableOp,classifier_graph_1/sequential_1/layer-1/bias*
_output_shapes
:2*
dtype0
?
-classifier_graph_1/sequential_1/output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2*>
shared_name/-classifier_graph_1/sequential_1/output/kernel
?
Aclassifier_graph_1/sequential_1/output/kernel/Read/ReadVariableOpReadVariableOp-classifier_graph_1/sequential_1/output/kernel*
_output_shapes

:2*
dtype0
?
+classifier_graph_1/sequential_1/output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*<
shared_name-+classifier_graph_1/sequential_1/output/bias
?
?classifier_graph_1/sequential_1/output/bias/Read/ReadVariableOpReadVariableOp+classifier_graph_1/sequential_1/output/bias*
_output_shapes
:*
dtype0
l
VariableVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_name
Variable
e
Variable/Read/ReadVariableOpReadVariableOpVariable*
_output_shapes

:*
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
trainable_variables
regularization_losses
	variables
	keras_api

signatures
 
i

Layers
		model

trainable_variables
regularization_losses
	variables
	keras_api

0
1
2
3
 
#
0
1
2
3
4
?
layer_regularization_losses
trainable_variables

layers
metrics
non_trainable_variables
regularization_losses
	variables
 

0
1
2
?
layer_with_weights-0
layer-0
layer-1
layer-2
trainable_variables
regularization_losses
	variables
	keras_api

0
1
2
3
 
#
0
1
2
3
4
?
layer_regularization_losses

trainable_variables

layers
 metrics
!non_trainable_variables
regularization_losses
	variables
tr
VARIABLE_VALUE.classifier_graph_1/sequential_1/layer-1/kernel0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE,classifier_graph_1/sequential_1/layer-1/bias0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE-classifier_graph_1/sequential_1/output/kernel0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUE+classifier_graph_1/sequential_1/output/bias0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUE
DB
VARIABLE_VALUEVariable&variables/4/.ATTRIBUTES/VARIABLE_VALUE
 

0
1
 

0
Y
w
"trainable_variables
#regularization_losses
$	variables
%	keras_api
h

kernel
bias
&trainable_variables
'regularization_losses
(	variables
)	keras_api
h

kernel
bias
*trainable_variables
+regularization_losses
,	variables
-	keras_api

0
1
2
3
 
#
0
1
2
3
4
?
.layer_regularization_losses
trainable_variables

/layers
0metrics
1non_trainable_variables
regularization_losses
	variables
 

0
1
2
	3
 

0
 
 

0
?
2layer_regularization_losses
"trainable_variables

3layers
4metrics
5non_trainable_variables
#regularization_losses
$	variables

0
1
 

0
1
?
6layer_regularization_losses
&trainable_variables

7layers
8metrics
9non_trainable_variables
'regularization_losses
(	variables

0
1
 

0
1
?
:layer_regularization_losses
*trainable_variables

;layers
<metrics
=non_trainable_variables
+regularization_losses
,	variables
 

0
1
2
 

0
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
z
serving_default_input_1Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1Variable.classifier_graph_1/sequential_1/layer-1/kernel,classifier_graph_1/sequential_1/layer-1/bias-classifier_graph_1/sequential_1/output/kernel+classifier_graph_1/sequential_1/output/bias*
Tin

2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????**
config_proto

CPU

GPU 2J 8*,
f'R%
#__inference_signature_wrapper_81345
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameBclassifier_graph_1/sequential_1/layer-1/kernel/Read/ReadVariableOp@classifier_graph_1/sequential_1/layer-1/bias/Read/ReadVariableOpAclassifier_graph_1/sequential_1/output/kernel/Read/ReadVariableOp?classifier_graph_1/sequential_1/output/bias/Read/ReadVariableOpVariable/Read/ReadVariableOpConst*
Tin
	2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*
_output_shapes
: **
config_proto

CPU

GPU 2J 8*'
f"R 
__inference__traced_save_81636
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename.classifier_graph_1/sequential_1/layer-1/kernel,classifier_graph_1/sequential_1/layer-1/bias-classifier_graph_1/sequential_1/output/kernel+classifier_graph_1/sequential_1/output/biasVariable*
Tin

2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*
_output_shapes
: **
config_proto

CPU

GPU 2J 8**
f%R#
!__inference__traced_restore_81663??
?
?
%__inference_model_layer_call_fn_81314
input_1"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1statefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5*
Tin

2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????**
config_proto

CPU

GPU 2J 8*I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_813062
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????:::::22
StatefulPartitionedCallStatefulPartitionedCall:' #
!
_user_specified_name	input_1
?6
?
@__inference_model_layer_call_and_return_conditional_losses_81397

inputsV
Rclassifier_graph_1_sequential_1_project_1_matrix_transpose_readvariableop_resourceJ
Fclassifier_graph_1_sequential_1_layer_1_matmul_readvariableop_resourceK
Gclassifier_graph_1_sequential_1_layer_1_biasadd_readvariableop_resourceI
Eclassifier_graph_1_sequential_1_output_matmul_readvariableop_resourceJ
Fclassifier_graph_1_sequential_1_output_biasadd_readvariableop_resource
identity??>classifier_graph_1/sequential_1/layer-1/BiasAdd/ReadVariableOp?=classifier_graph_1/sequential_1/layer-1/MatMul/ReadVariableOp?=classifier_graph_1/sequential_1/output/BiasAdd/ReadVariableOp?<classifier_graph_1/sequential_1/output/MatMul/ReadVariableOp?Aclassifier_graph_1/sequential_1/project_1/matmul_1/ReadVariableOp?Iclassifier_graph_1/sequential_1/project_1/matrix_transpose/ReadVariableOp?
Iclassifier_graph_1/sequential_1/project_1/matrix_transpose/ReadVariableOpReadVariableOpRclassifier_graph_1_sequential_1_project_1_matrix_transpose_readvariableop_resource*
_output_shapes

:*
dtype02K
Iclassifier_graph_1/sequential_1/project_1/matrix_transpose/ReadVariableOp?
Iclassifier_graph_1/sequential_1/project_1/matrix_transpose/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2K
Iclassifier_graph_1/sequential_1/project_1/matrix_transpose/transpose/perm?
Dclassifier_graph_1/sequential_1/project_1/matrix_transpose/transpose	TransposeQclassifier_graph_1/sequential_1/project_1/matrix_transpose/ReadVariableOp:value:0Rclassifier_graph_1/sequential_1/project_1/matrix_transpose/transpose/perm:output:0*
T0*
_output_shapes

:2F
Dclassifier_graph_1/sequential_1/project_1/matrix_transpose/transpose?
0classifier_graph_1/sequential_1/project_1/matmulMatMulinputsHclassifier_graph_1/sequential_1/project_1/matrix_transpose/transpose:y:0*
T0*'
_output_shapes
:?????????22
0classifier_graph_1/sequential_1/project_1/matmul?
Aclassifier_graph_1/sequential_1/project_1/matmul_1/ReadVariableOpReadVariableOpRclassifier_graph_1_sequential_1_project_1_matrix_transpose_readvariableop_resourceJ^classifier_graph_1/sequential_1/project_1/matrix_transpose/ReadVariableOp*
_output_shapes

:*
dtype02C
Aclassifier_graph_1/sequential_1/project_1/matmul_1/ReadVariableOp?
2classifier_graph_1/sequential_1/project_1/matmul_1MatMul:classifier_graph_1/sequential_1/project_1/matmul:product:0Iclassifier_graph_1/sequential_1/project_1/matmul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????24
2classifier_graph_1/sequential_1/project_1/matmul_1?
-classifier_graph_1/sequential_1/project_1/subSubinputs<classifier_graph_1/sequential_1/project_1/matmul_1:product:0*
T0*'
_output_shapes
:?????????2/
-classifier_graph_1/sequential_1/project_1/sub?
=classifier_graph_1/sequential_1/layer-1/MatMul/ReadVariableOpReadVariableOpFclassifier_graph_1_sequential_1_layer_1_matmul_readvariableop_resource*
_output_shapes

:2*
dtype02?
=classifier_graph_1/sequential_1/layer-1/MatMul/ReadVariableOp?
.classifier_graph_1/sequential_1/layer-1/MatMulMatMul1classifier_graph_1/sequential_1/project_1/sub:z:0Eclassifier_graph_1/sequential_1/layer-1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????220
.classifier_graph_1/sequential_1/layer-1/MatMul?
>classifier_graph_1/sequential_1/layer-1/BiasAdd/ReadVariableOpReadVariableOpGclassifier_graph_1_sequential_1_layer_1_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02@
>classifier_graph_1/sequential_1/layer-1/BiasAdd/ReadVariableOp?
/classifier_graph_1/sequential_1/layer-1/BiasAddBiasAdd8classifier_graph_1/sequential_1/layer-1/MatMul:product:0Fclassifier_graph_1/sequential_1/layer-1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????221
/classifier_graph_1/sequential_1/layer-1/BiasAdd?
,classifier_graph_1/sequential_1/layer-1/ReluRelu8classifier_graph_1/sequential_1/layer-1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????22.
,classifier_graph_1/sequential_1/layer-1/Relu?
<classifier_graph_1/sequential_1/output/MatMul/ReadVariableOpReadVariableOpEclassifier_graph_1_sequential_1_output_matmul_readvariableop_resource*
_output_shapes

:2*
dtype02>
<classifier_graph_1/sequential_1/output/MatMul/ReadVariableOp?
-classifier_graph_1/sequential_1/output/MatMulMatMul:classifier_graph_1/sequential_1/layer-1/Relu:activations:0Dclassifier_graph_1/sequential_1/output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2/
-classifier_graph_1/sequential_1/output/MatMul?
=classifier_graph_1/sequential_1/output/BiasAdd/ReadVariableOpReadVariableOpFclassifier_graph_1_sequential_1_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02?
=classifier_graph_1/sequential_1/output/BiasAdd/ReadVariableOp?
.classifier_graph_1/sequential_1/output/BiasAddBiasAdd7classifier_graph_1/sequential_1/output/MatMul:product:0Eclassifier_graph_1/sequential_1/output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????20
.classifier_graph_1/sequential_1/output/BiasAdd?
.classifier_graph_1/sequential_1/output/SoftmaxSoftmax7classifier_graph_1/sequential_1/output/BiasAdd:output:0*
T0*'
_output_shapes
:?????????20
.classifier_graph_1/sequential_1/output/Softmax?
IdentityIdentity8classifier_graph_1/sequential_1/output/Softmax:softmax:0?^classifier_graph_1/sequential_1/layer-1/BiasAdd/ReadVariableOp>^classifier_graph_1/sequential_1/layer-1/MatMul/ReadVariableOp>^classifier_graph_1/sequential_1/output/BiasAdd/ReadVariableOp=^classifier_graph_1/sequential_1/output/MatMul/ReadVariableOpB^classifier_graph_1/sequential_1/project_1/matmul_1/ReadVariableOpJ^classifier_graph_1/sequential_1/project_1/matrix_transpose/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????:::::2?
>classifier_graph_1/sequential_1/layer-1/BiasAdd/ReadVariableOp>classifier_graph_1/sequential_1/layer-1/BiasAdd/ReadVariableOp2~
=classifier_graph_1/sequential_1/layer-1/MatMul/ReadVariableOp=classifier_graph_1/sequential_1/layer-1/MatMul/ReadVariableOp2~
=classifier_graph_1/sequential_1/output/BiasAdd/ReadVariableOp=classifier_graph_1/sequential_1/output/BiasAdd/ReadVariableOp2|
<classifier_graph_1/sequential_1/output/MatMul/ReadVariableOp<classifier_graph_1/sequential_1/output/MatMul/ReadVariableOp2?
Aclassifier_graph_1/sequential_1/project_1/matmul_1/ReadVariableOpAclassifier_graph_1/sequential_1/project_1/matmul_1/ReadVariableOp2?
Iclassifier_graph_1/sequential_1/project_1/matrix_transpose/ReadVariableOpIclassifier_graph_1/sequential_1/project_1/matrix_transpose/ReadVariableOp:& "
 
_user_specified_nameinputs
?
?
'__inference_layer-1_layer_call_fn_81579

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????2**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_layer-1_layer_call_and_return_conditional_losses_810612
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
?
?
,__inference_sequential_1_layer_call_fn_81154
input_1"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1statefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5*
Tin

2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????**
config_proto

CPU

GPU 2J 8*P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_811462
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????:::::22
StatefulPartitionedCallStatefulPartitionedCall:' #
!
_user_specified_name	input_1
?	
?
A__inference_output_layer_call_and_return_conditional_losses_81590

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
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
?
?
,__inference_sequential_1_layer_call_fn_81132
input_1"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1statefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5*
Tin

2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????**
config_proto

CPU

GPU 2J 8*P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_811242
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????:::::22
StatefulPartitionedCallStatefulPartitionedCall:' #
!
_user_specified_name	input_1
?
?
)__inference_project_1_layer_call_fn_81044
x"
statefulpartitionedcall_args_1
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallxstatefulpartitionedcall_args_1*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????**
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_project_1_layer_call_and_return_conditional_losses_810372
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:22
StatefulPartitionedCallStatefulPartitionedCall:! 

_user_specified_namex
?
?
__inference__traced_save_81636
file_prefixM
Isavev2_classifier_graph_1_sequential_1_layer_1_kernel_read_readvariableopK
Gsavev2_classifier_graph_1_sequential_1_layer_1_bias_read_readvariableopL
Hsavev2_classifier_graph_1_sequential_1_output_kernel_read_readvariableopJ
Fsavev2_classifier_graph_1_sequential_1_output_bias_read_readvariableop'
#savev2_variable_read_readvariableop
savev2_1_const

identity_1??MergeV2Checkpoints?SaveV2?SaveV2_1?
StringJoin/inputs_1Const"/device:CPU:0*
_output_shapes
: *
dtype0*<
value3B1 B+_temp_1e6bad1c9171428e9ae338e69bda38a4/part2
StringJoin/inputs_1?

StringJoin
StringJoinfile_prefixStringJoin/inputs_1:output:0"/device:CPU:0*
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

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
:*
dtype0*?
value?B?B0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUE2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Isavev2_classifier_graph_1_sequential_1_layer_1_kernel_read_readvariableopGsavev2_classifier_graph_1_sequential_1_layer_1_bias_read_readvariableopHsavev2_classifier_graph_1_sequential_1_output_kernel_read_readvariableopFsavev2_classifier_graph_1_sequential_1_output_bias_read_readvariableop#savev2_variable_read_readvariableop"/device:CPU:0*
_output_shapes
 *
dtypes	
22
SaveV2?
ShardedFilename_1/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B :2
ShardedFilename_1/shard?
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename_1?
SaveV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2_1/tensor_names?
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
SaveV2_1/shape_and_slices?
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
_output_shapes
 *
dtypes
22

SaveV2_1?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity?

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*A
_input_shapes0
.: :2:2:2::: 2(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV22
SaveV2_1SaveV2_1:+ '
%
_user_specified_namefile_prefix
?	
?
B__inference_layer-1_layer_call_and_return_conditional_losses_81572

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:2*
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
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
?

?
M__inference_classifier_graph_1_layer_call_and_return_conditional_losses_81208
x/
+sequential_1_statefulpartitionedcall_args_1/
+sequential_1_statefulpartitionedcall_args_2/
+sequential_1_statefulpartitionedcall_args_3/
+sequential_1_statefulpartitionedcall_args_4/
+sequential_1_statefulpartitionedcall_args_5
identity??$sequential_1/StatefulPartitionedCall?
$sequential_1/StatefulPartitionedCallStatefulPartitionedCallx+sequential_1_statefulpartitionedcall_args_1+sequential_1_statefulpartitionedcall_args_2+sequential_1_statefulpartitionedcall_args_3+sequential_1_statefulpartitionedcall_args_4+sequential_1_statefulpartitionedcall_args_5*
Tin

2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????**
config_proto

CPU

GPU 2J 8*P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_811462&
$sequential_1/StatefulPartitionedCall?
IdentityIdentity-sequential_1/StatefulPartitionedCall:output:0%^sequential_1/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????:::::2L
$sequential_1/StatefulPartitionedCall$sequential_1/StatefulPartitionedCall:! 

_user_specified_namex
?!
?
G__inference_sequential_1_layer_call_and_return_conditional_losses_81515

inputs6
2project_1_matrix_transpose_readvariableop_resource*
&layer_1_matmul_readvariableop_resource+
'layer_1_biasadd_readvariableop_resource)
%output_matmul_readvariableop_resource*
&output_biasadd_readvariableop_resource
identity??layer-1/BiasAdd/ReadVariableOp?layer-1/MatMul/ReadVariableOp?output/BiasAdd/ReadVariableOp?output/MatMul/ReadVariableOp?!project_1/matmul_1/ReadVariableOp?)project_1/matrix_transpose/ReadVariableOp?
)project_1/matrix_transpose/ReadVariableOpReadVariableOp2project_1_matrix_transpose_readvariableop_resource*
_output_shapes

:*
dtype02+
)project_1/matrix_transpose/ReadVariableOp?
)project_1/matrix_transpose/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2+
)project_1/matrix_transpose/transpose/perm?
$project_1/matrix_transpose/transpose	Transpose1project_1/matrix_transpose/ReadVariableOp:value:02project_1/matrix_transpose/transpose/perm:output:0*
T0*
_output_shapes

:2&
$project_1/matrix_transpose/transpose?
project_1/matmulMatMulinputs(project_1/matrix_transpose/transpose:y:0*
T0*'
_output_shapes
:?????????2
project_1/matmul?
!project_1/matmul_1/ReadVariableOpReadVariableOp2project_1_matrix_transpose_readvariableop_resource*^project_1/matrix_transpose/ReadVariableOp*
_output_shapes

:*
dtype02#
!project_1/matmul_1/ReadVariableOp?
project_1/matmul_1MatMulproject_1/matmul:product:0)project_1/matmul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
project_1/matmul_1}
project_1/subSubinputsproject_1/matmul_1:product:0*
T0*'
_output_shapes
:?????????2
project_1/sub?
layer-1/MatMul/ReadVariableOpReadVariableOp&layer_1_matmul_readvariableop_resource*
_output_shapes

:2*
dtype02
layer-1/MatMul/ReadVariableOp?
layer-1/MatMulMatMulproject_1/sub:z:0%layer-1/MatMul/ReadVariableOp:value:0*
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
IdentityIdentityoutput/Softmax:softmax:0^layer-1/BiasAdd/ReadVariableOp^layer-1/MatMul/ReadVariableOp^output/BiasAdd/ReadVariableOp^output/MatMul/ReadVariableOp"^project_1/matmul_1/ReadVariableOp*^project_1/matrix_transpose/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????:::::2@
layer-1/BiasAdd/ReadVariableOplayer-1/BiasAdd/ReadVariableOp2>
layer-1/MatMul/ReadVariableOplayer-1/MatMul/ReadVariableOp2>
output/BiasAdd/ReadVariableOpoutput/BiasAdd/ReadVariableOp2<
output/MatMul/ReadVariableOpoutput/MatMul/ReadVariableOp2F
!project_1/matmul_1/ReadVariableOp!project_1/matmul_1/ReadVariableOp2V
)project_1/matrix_transpose/ReadVariableOp)project_1/matrix_transpose/ReadVariableOp:& "
 
_user_specified_nameinputs
?
?
%__inference_model_layer_call_fn_81417

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5*
Tin

2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????**
config_proto

CPU

GPU 2J 8*I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_813262
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????:::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
?
?
,__inference_sequential_1_layer_call_fn_81561

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5*
Tin

2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????**
config_proto

CPU

GPU 2J 8*P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_811462
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????:::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
?)
?
M__inference_classifier_graph_1_layer_call_and_return_conditional_losses_81469
xC
?sequential_1_project_1_matrix_transpose_readvariableop_resource7
3sequential_1_layer_1_matmul_readvariableop_resource8
4sequential_1_layer_1_biasadd_readvariableop_resource6
2sequential_1_output_matmul_readvariableop_resource7
3sequential_1_output_biasadd_readvariableop_resource
identity??+sequential_1/layer-1/BiasAdd/ReadVariableOp?*sequential_1/layer-1/MatMul/ReadVariableOp?*sequential_1/output/BiasAdd/ReadVariableOp?)sequential_1/output/MatMul/ReadVariableOp?.sequential_1/project_1/matmul_1/ReadVariableOp?6sequential_1/project_1/matrix_transpose/ReadVariableOp?
6sequential_1/project_1/matrix_transpose/ReadVariableOpReadVariableOp?sequential_1_project_1_matrix_transpose_readvariableop_resource*
_output_shapes

:*
dtype028
6sequential_1/project_1/matrix_transpose/ReadVariableOp?
6sequential_1/project_1/matrix_transpose/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       28
6sequential_1/project_1/matrix_transpose/transpose/perm?
1sequential_1/project_1/matrix_transpose/transpose	Transpose>sequential_1/project_1/matrix_transpose/ReadVariableOp:value:0?sequential_1/project_1/matrix_transpose/transpose/perm:output:0*
T0*
_output_shapes

:23
1sequential_1/project_1/matrix_transpose/transpose?
sequential_1/project_1/matmulMatMulx5sequential_1/project_1/matrix_transpose/transpose:y:0*
T0*'
_output_shapes
:?????????2
sequential_1/project_1/matmul?
.sequential_1/project_1/matmul_1/ReadVariableOpReadVariableOp?sequential_1_project_1_matrix_transpose_readvariableop_resource7^sequential_1/project_1/matrix_transpose/ReadVariableOp*
_output_shapes

:*
dtype020
.sequential_1/project_1/matmul_1/ReadVariableOp?
sequential_1/project_1/matmul_1MatMul'sequential_1/project_1/matmul:product:06sequential_1/project_1/matmul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2!
sequential_1/project_1/matmul_1?
sequential_1/project_1/subSubx)sequential_1/project_1/matmul_1:product:0*
T0*'
_output_shapes
:?????????2
sequential_1/project_1/sub?
*sequential_1/layer-1/MatMul/ReadVariableOpReadVariableOp3sequential_1_layer_1_matmul_readvariableop_resource*
_output_shapes

:2*
dtype02,
*sequential_1/layer-1/MatMul/ReadVariableOp?
sequential_1/layer-1/MatMulMatMulsequential_1/project_1/sub:z:02sequential_1/layer-1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
sequential_1/layer-1/MatMul?
+sequential_1/layer-1/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_layer_1_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02-
+sequential_1/layer-1/BiasAdd/ReadVariableOp?
sequential_1/layer-1/BiasAddBiasAdd%sequential_1/layer-1/MatMul:product:03sequential_1/layer-1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
sequential_1/layer-1/BiasAdd?
sequential_1/layer-1/ReluRelu%sequential_1/layer-1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????22
sequential_1/layer-1/Relu?
)sequential_1/output/MatMul/ReadVariableOpReadVariableOp2sequential_1_output_matmul_readvariableop_resource*
_output_shapes

:2*
dtype02+
)sequential_1/output/MatMul/ReadVariableOp?
sequential_1/output/MatMulMatMul'sequential_1/layer-1/Relu:activations:01sequential_1/output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential_1/output/MatMul?
*sequential_1/output/BiasAdd/ReadVariableOpReadVariableOp3sequential_1_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02,
*sequential_1/output/BiasAdd/ReadVariableOp?
sequential_1/output/BiasAddBiasAdd$sequential_1/output/MatMul:product:02sequential_1/output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential_1/output/BiasAdd?
sequential_1/output/SoftmaxSoftmax$sequential_1/output/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
sequential_1/output/Softmax?
IdentityIdentity%sequential_1/output/Softmax:softmax:0,^sequential_1/layer-1/BiasAdd/ReadVariableOp+^sequential_1/layer-1/MatMul/ReadVariableOp+^sequential_1/output/BiasAdd/ReadVariableOp*^sequential_1/output/MatMul/ReadVariableOp/^sequential_1/project_1/matmul_1/ReadVariableOp7^sequential_1/project_1/matrix_transpose/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????:::::2Z
+sequential_1/layer-1/BiasAdd/ReadVariableOp+sequential_1/layer-1/BiasAdd/ReadVariableOp2X
*sequential_1/layer-1/MatMul/ReadVariableOp*sequential_1/layer-1/MatMul/ReadVariableOp2X
*sequential_1/output/BiasAdd/ReadVariableOp*sequential_1/output/BiasAdd/ReadVariableOp2V
)sequential_1/output/MatMul/ReadVariableOp)sequential_1/output/MatMul/ReadVariableOp2`
.sequential_1/project_1/matmul_1/ReadVariableOp.sequential_1/project_1/matmul_1/ReadVariableOp2p
6sequential_1/project_1/matrix_transpose/ReadVariableOp6sequential_1/project_1/matrix_transpose/ReadVariableOp:! 

_user_specified_namex
?	
?
B__inference_layer-1_layer_call_and_return_conditional_losses_81061

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:2*
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
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
?
?
!__inference__traced_restore_81663
file_prefixC
?assignvariableop_classifier_graph_1_sequential_1_layer_1_kernelC
?assignvariableop_1_classifier_graph_1_sequential_1_layer_1_biasD
@assignvariableop_2_classifier_graph_1_sequential_1_output_kernelB
>assignvariableop_3_classifier_graph_1_sequential_1_output_bias
assignvariableop_4_variable

identity_6??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_2?AssignVariableOp_3?AssignVariableOp_4?	RestoreV2?RestoreV2_1?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUE2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*(
_output_shapes
:::::*
dtypes	
22
	RestoreV2X
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp?assignvariableop_classifier_graph_1_sequential_1_layer_1_kernelIdentity:output:0*
_output_shapes
 *
dtype02
AssignVariableOp\

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp?assignvariableop_1_classifier_graph_1_sequential_1_layer_1_biasIdentity_1:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_1\

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp@assignvariableop_2_classifier_graph_1_sequential_1_output_kernelIdentity_2:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_2\

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp>assignvariableop_3_classifier_graph_1_sequential_1_output_biasIdentity_3:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_3\

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOpassignvariableop_4_variableIdentity_4:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_4?
RestoreV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2_1/tensor_names?
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
RestoreV2_1/shape_and_slices?
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
_output_shapes
:*
dtypes
22
RestoreV2_19
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?

Identity_5Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_5?

Identity_6IdentityIdentity_5:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4
^RestoreV2^RestoreV2_1*
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
AssignVariableOp_4AssignVariableOp_42
	RestoreV2	RestoreV22
RestoreV2_1RestoreV2_1:+ '
%
_user_specified_namefile_prefix
?
?
G__inference_sequential_1_layer_call_and_return_conditional_losses_81097
input_1,
(project_1_statefulpartitionedcall_args_1*
&layer_1_statefulpartitionedcall_args_1*
&layer_1_statefulpartitionedcall_args_2)
%output_statefulpartitionedcall_args_1)
%output_statefulpartitionedcall_args_2
identity??layer-1/StatefulPartitionedCall?output/StatefulPartitionedCall?!project_1/StatefulPartitionedCall?
!project_1/StatefulPartitionedCallStatefulPartitionedCallinput_1(project_1_statefulpartitionedcall_args_1*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????**
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_project_1_layer_call_and_return_conditional_losses_810372#
!project_1/StatefulPartitionedCall?
layer-1/StatefulPartitionedCallStatefulPartitionedCall*project_1/StatefulPartitionedCall:output:0&layer_1_statefulpartitionedcall_args_1&layer_1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????2**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_layer-1_layer_call_and_return_conditional_losses_810612!
layer-1/StatefulPartitionedCall?
output/StatefulPartitionedCallStatefulPartitionedCall(layer-1/StatefulPartitionedCall:output:0%output_statefulpartitionedcall_args_1%output_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????**
config_proto

CPU

GPU 2J 8*J
fERC
A__inference_output_layer_call_and_return_conditional_losses_810842 
output/StatefulPartitionedCall?
IdentityIdentity'output/StatefulPartitionedCall:output:0 ^layer-1/StatefulPartitionedCall^output/StatefulPartitionedCall"^project_1/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????:::::2B
layer-1/StatefulPartitionedCalllayer-1/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall2F
!project_1/StatefulPartitionedCall!project_1/StatefulPartitionedCall:' #
!
_user_specified_name	input_1
?

?
M__inference_classifier_graph_1_layer_call_and_return_conditional_losses_81195
input_1/
+sequential_1_statefulpartitionedcall_args_1/
+sequential_1_statefulpartitionedcall_args_2/
+sequential_1_statefulpartitionedcall_args_3/
+sequential_1_statefulpartitionedcall_args_4/
+sequential_1_statefulpartitionedcall_args_5
identity??$sequential_1/StatefulPartitionedCall?
$sequential_1/StatefulPartitionedCallStatefulPartitionedCallinput_1+sequential_1_statefulpartitionedcall_args_1+sequential_1_statefulpartitionedcall_args_2+sequential_1_statefulpartitionedcall_args_3+sequential_1_statefulpartitionedcall_args_4+sequential_1_statefulpartitionedcall_args_5*
Tin

2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????**
config_proto

CPU

GPU 2J 8*P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_811462&
$sequential_1/StatefulPartitionedCall?
IdentityIdentity-sequential_1/StatefulPartitionedCall:output:0%^sequential_1/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????:::::2L
$sequential_1/StatefulPartitionedCall$sequential_1/StatefulPartitionedCall:' #
!
_user_specified_name	input_1
?
?
@__inference_model_layer_call_and_return_conditional_losses_81293
input_15
1classifier_graph_1_statefulpartitionedcall_args_15
1classifier_graph_1_statefulpartitionedcall_args_25
1classifier_graph_1_statefulpartitionedcall_args_35
1classifier_graph_1_statefulpartitionedcall_args_45
1classifier_graph_1_statefulpartitionedcall_args_5
identity??*classifier_graph_1/StatefulPartitionedCall?
*classifier_graph_1/StatefulPartitionedCallStatefulPartitionedCallinput_11classifier_graph_1_statefulpartitionedcall_args_11classifier_graph_1_statefulpartitionedcall_args_21classifier_graph_1_statefulpartitionedcall_args_31classifier_graph_1_statefulpartitionedcall_args_41classifier_graph_1_statefulpartitionedcall_args_5*
Tin

2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????**
config_proto

CPU

GPU 2J 8*V
fQRO
M__inference_classifier_graph_1_layer_call_and_return_conditional_losses_812082,
*classifier_graph_1/StatefulPartitionedCall?
IdentityIdentity3classifier_graph_1/StatefulPartitionedCall:output:0+^classifier_graph_1/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????:::::2X
*classifier_graph_1/StatefulPartitionedCall*classifier_graph_1/StatefulPartitionedCall:' #
!
_user_specified_name	input_1
?:
?
 __inference__wrapped_model_81024
input_1\
Xmodel_classifier_graph_1_sequential_1_project_1_matrix_transpose_readvariableop_resourceP
Lmodel_classifier_graph_1_sequential_1_layer_1_matmul_readvariableop_resourceQ
Mmodel_classifier_graph_1_sequential_1_layer_1_biasadd_readvariableop_resourceO
Kmodel_classifier_graph_1_sequential_1_output_matmul_readvariableop_resourceP
Lmodel_classifier_graph_1_sequential_1_output_biasadd_readvariableop_resource
identity??Dmodel/classifier_graph_1/sequential_1/layer-1/BiasAdd/ReadVariableOp?Cmodel/classifier_graph_1/sequential_1/layer-1/MatMul/ReadVariableOp?Cmodel/classifier_graph_1/sequential_1/output/BiasAdd/ReadVariableOp?Bmodel/classifier_graph_1/sequential_1/output/MatMul/ReadVariableOp?Gmodel/classifier_graph_1/sequential_1/project_1/matmul_1/ReadVariableOp?Omodel/classifier_graph_1/sequential_1/project_1/matrix_transpose/ReadVariableOp?
Omodel/classifier_graph_1/sequential_1/project_1/matrix_transpose/ReadVariableOpReadVariableOpXmodel_classifier_graph_1_sequential_1_project_1_matrix_transpose_readvariableop_resource*
_output_shapes

:*
dtype02Q
Omodel/classifier_graph_1/sequential_1/project_1/matrix_transpose/ReadVariableOp?
Omodel/classifier_graph_1/sequential_1/project_1/matrix_transpose/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2Q
Omodel/classifier_graph_1/sequential_1/project_1/matrix_transpose/transpose/perm?
Jmodel/classifier_graph_1/sequential_1/project_1/matrix_transpose/transpose	TransposeWmodel/classifier_graph_1/sequential_1/project_1/matrix_transpose/ReadVariableOp:value:0Xmodel/classifier_graph_1/sequential_1/project_1/matrix_transpose/transpose/perm:output:0*
T0*
_output_shapes

:2L
Jmodel/classifier_graph_1/sequential_1/project_1/matrix_transpose/transpose?
6model/classifier_graph_1/sequential_1/project_1/matmulMatMulinput_1Nmodel/classifier_graph_1/sequential_1/project_1/matrix_transpose/transpose:y:0*
T0*'
_output_shapes
:?????????28
6model/classifier_graph_1/sequential_1/project_1/matmul?
Gmodel/classifier_graph_1/sequential_1/project_1/matmul_1/ReadVariableOpReadVariableOpXmodel_classifier_graph_1_sequential_1_project_1_matrix_transpose_readvariableop_resourceP^model/classifier_graph_1/sequential_1/project_1/matrix_transpose/ReadVariableOp*
_output_shapes

:*
dtype02I
Gmodel/classifier_graph_1/sequential_1/project_1/matmul_1/ReadVariableOp?
8model/classifier_graph_1/sequential_1/project_1/matmul_1MatMul@model/classifier_graph_1/sequential_1/project_1/matmul:product:0Omodel/classifier_graph_1/sequential_1/project_1/matmul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2:
8model/classifier_graph_1/sequential_1/project_1/matmul_1?
3model/classifier_graph_1/sequential_1/project_1/subSubinput_1Bmodel/classifier_graph_1/sequential_1/project_1/matmul_1:product:0*
T0*'
_output_shapes
:?????????25
3model/classifier_graph_1/sequential_1/project_1/sub?
Cmodel/classifier_graph_1/sequential_1/layer-1/MatMul/ReadVariableOpReadVariableOpLmodel_classifier_graph_1_sequential_1_layer_1_matmul_readvariableop_resource*
_output_shapes

:2*
dtype02E
Cmodel/classifier_graph_1/sequential_1/layer-1/MatMul/ReadVariableOp?
4model/classifier_graph_1/sequential_1/layer-1/MatMulMatMul7model/classifier_graph_1/sequential_1/project_1/sub:z:0Kmodel/classifier_graph_1/sequential_1/layer-1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????226
4model/classifier_graph_1/sequential_1/layer-1/MatMul?
Dmodel/classifier_graph_1/sequential_1/layer-1/BiasAdd/ReadVariableOpReadVariableOpMmodel_classifier_graph_1_sequential_1_layer_1_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02F
Dmodel/classifier_graph_1/sequential_1/layer-1/BiasAdd/ReadVariableOp?
5model/classifier_graph_1/sequential_1/layer-1/BiasAddBiasAdd>model/classifier_graph_1/sequential_1/layer-1/MatMul:product:0Lmodel/classifier_graph_1/sequential_1/layer-1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????227
5model/classifier_graph_1/sequential_1/layer-1/BiasAdd?
2model/classifier_graph_1/sequential_1/layer-1/ReluRelu>model/classifier_graph_1/sequential_1/layer-1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????224
2model/classifier_graph_1/sequential_1/layer-1/Relu?
Bmodel/classifier_graph_1/sequential_1/output/MatMul/ReadVariableOpReadVariableOpKmodel_classifier_graph_1_sequential_1_output_matmul_readvariableop_resource*
_output_shapes

:2*
dtype02D
Bmodel/classifier_graph_1/sequential_1/output/MatMul/ReadVariableOp?
3model/classifier_graph_1/sequential_1/output/MatMulMatMul@model/classifier_graph_1/sequential_1/layer-1/Relu:activations:0Jmodel/classifier_graph_1/sequential_1/output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????25
3model/classifier_graph_1/sequential_1/output/MatMul?
Cmodel/classifier_graph_1/sequential_1/output/BiasAdd/ReadVariableOpReadVariableOpLmodel_classifier_graph_1_sequential_1_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02E
Cmodel/classifier_graph_1/sequential_1/output/BiasAdd/ReadVariableOp?
4model/classifier_graph_1/sequential_1/output/BiasAddBiasAdd=model/classifier_graph_1/sequential_1/output/MatMul:product:0Kmodel/classifier_graph_1/sequential_1/output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????26
4model/classifier_graph_1/sequential_1/output/BiasAdd?
4model/classifier_graph_1/sequential_1/output/SoftmaxSoftmax=model/classifier_graph_1/sequential_1/output/BiasAdd:output:0*
T0*'
_output_shapes
:?????????26
4model/classifier_graph_1/sequential_1/output/Softmax?
IdentityIdentity>model/classifier_graph_1/sequential_1/output/Softmax:softmax:0E^model/classifier_graph_1/sequential_1/layer-1/BiasAdd/ReadVariableOpD^model/classifier_graph_1/sequential_1/layer-1/MatMul/ReadVariableOpD^model/classifier_graph_1/sequential_1/output/BiasAdd/ReadVariableOpC^model/classifier_graph_1/sequential_1/output/MatMul/ReadVariableOpH^model/classifier_graph_1/sequential_1/project_1/matmul_1/ReadVariableOpP^model/classifier_graph_1/sequential_1/project_1/matrix_transpose/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????:::::2?
Dmodel/classifier_graph_1/sequential_1/layer-1/BiasAdd/ReadVariableOpDmodel/classifier_graph_1/sequential_1/layer-1/BiasAdd/ReadVariableOp2?
Cmodel/classifier_graph_1/sequential_1/layer-1/MatMul/ReadVariableOpCmodel/classifier_graph_1/sequential_1/layer-1/MatMul/ReadVariableOp2?
Cmodel/classifier_graph_1/sequential_1/output/BiasAdd/ReadVariableOpCmodel/classifier_graph_1/sequential_1/output/BiasAdd/ReadVariableOp2?
Bmodel/classifier_graph_1/sequential_1/output/MatMul/ReadVariableOpBmodel/classifier_graph_1/sequential_1/output/MatMul/ReadVariableOp2?
Gmodel/classifier_graph_1/sequential_1/project_1/matmul_1/ReadVariableOpGmodel/classifier_graph_1/sequential_1/project_1/matmul_1/ReadVariableOp2?
Omodel/classifier_graph_1/sequential_1/project_1/matrix_transpose/ReadVariableOpOmodel/classifier_graph_1/sequential_1/project_1/matrix_transpose/ReadVariableOp:' #
!
_user_specified_name	input_1
?6
?
@__inference_model_layer_call_and_return_conditional_losses_81371

inputsV
Rclassifier_graph_1_sequential_1_project_1_matrix_transpose_readvariableop_resourceJ
Fclassifier_graph_1_sequential_1_layer_1_matmul_readvariableop_resourceK
Gclassifier_graph_1_sequential_1_layer_1_biasadd_readvariableop_resourceI
Eclassifier_graph_1_sequential_1_output_matmul_readvariableop_resourceJ
Fclassifier_graph_1_sequential_1_output_biasadd_readvariableop_resource
identity??>classifier_graph_1/sequential_1/layer-1/BiasAdd/ReadVariableOp?=classifier_graph_1/sequential_1/layer-1/MatMul/ReadVariableOp?=classifier_graph_1/sequential_1/output/BiasAdd/ReadVariableOp?<classifier_graph_1/sequential_1/output/MatMul/ReadVariableOp?Aclassifier_graph_1/sequential_1/project_1/matmul_1/ReadVariableOp?Iclassifier_graph_1/sequential_1/project_1/matrix_transpose/ReadVariableOp?
Iclassifier_graph_1/sequential_1/project_1/matrix_transpose/ReadVariableOpReadVariableOpRclassifier_graph_1_sequential_1_project_1_matrix_transpose_readvariableop_resource*
_output_shapes

:*
dtype02K
Iclassifier_graph_1/sequential_1/project_1/matrix_transpose/ReadVariableOp?
Iclassifier_graph_1/sequential_1/project_1/matrix_transpose/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2K
Iclassifier_graph_1/sequential_1/project_1/matrix_transpose/transpose/perm?
Dclassifier_graph_1/sequential_1/project_1/matrix_transpose/transpose	TransposeQclassifier_graph_1/sequential_1/project_1/matrix_transpose/ReadVariableOp:value:0Rclassifier_graph_1/sequential_1/project_1/matrix_transpose/transpose/perm:output:0*
T0*
_output_shapes

:2F
Dclassifier_graph_1/sequential_1/project_1/matrix_transpose/transpose?
0classifier_graph_1/sequential_1/project_1/matmulMatMulinputsHclassifier_graph_1/sequential_1/project_1/matrix_transpose/transpose:y:0*
T0*'
_output_shapes
:?????????22
0classifier_graph_1/sequential_1/project_1/matmul?
Aclassifier_graph_1/sequential_1/project_1/matmul_1/ReadVariableOpReadVariableOpRclassifier_graph_1_sequential_1_project_1_matrix_transpose_readvariableop_resourceJ^classifier_graph_1/sequential_1/project_1/matrix_transpose/ReadVariableOp*
_output_shapes

:*
dtype02C
Aclassifier_graph_1/sequential_1/project_1/matmul_1/ReadVariableOp?
2classifier_graph_1/sequential_1/project_1/matmul_1MatMul:classifier_graph_1/sequential_1/project_1/matmul:product:0Iclassifier_graph_1/sequential_1/project_1/matmul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????24
2classifier_graph_1/sequential_1/project_1/matmul_1?
-classifier_graph_1/sequential_1/project_1/subSubinputs<classifier_graph_1/sequential_1/project_1/matmul_1:product:0*
T0*'
_output_shapes
:?????????2/
-classifier_graph_1/sequential_1/project_1/sub?
=classifier_graph_1/sequential_1/layer-1/MatMul/ReadVariableOpReadVariableOpFclassifier_graph_1_sequential_1_layer_1_matmul_readvariableop_resource*
_output_shapes

:2*
dtype02?
=classifier_graph_1/sequential_1/layer-1/MatMul/ReadVariableOp?
.classifier_graph_1/sequential_1/layer-1/MatMulMatMul1classifier_graph_1/sequential_1/project_1/sub:z:0Eclassifier_graph_1/sequential_1/layer-1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????220
.classifier_graph_1/sequential_1/layer-1/MatMul?
>classifier_graph_1/sequential_1/layer-1/BiasAdd/ReadVariableOpReadVariableOpGclassifier_graph_1_sequential_1_layer_1_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02@
>classifier_graph_1/sequential_1/layer-1/BiasAdd/ReadVariableOp?
/classifier_graph_1/sequential_1/layer-1/BiasAddBiasAdd8classifier_graph_1/sequential_1/layer-1/MatMul:product:0Fclassifier_graph_1/sequential_1/layer-1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????221
/classifier_graph_1/sequential_1/layer-1/BiasAdd?
,classifier_graph_1/sequential_1/layer-1/ReluRelu8classifier_graph_1/sequential_1/layer-1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????22.
,classifier_graph_1/sequential_1/layer-1/Relu?
<classifier_graph_1/sequential_1/output/MatMul/ReadVariableOpReadVariableOpEclassifier_graph_1_sequential_1_output_matmul_readvariableop_resource*
_output_shapes

:2*
dtype02>
<classifier_graph_1/sequential_1/output/MatMul/ReadVariableOp?
-classifier_graph_1/sequential_1/output/MatMulMatMul:classifier_graph_1/sequential_1/layer-1/Relu:activations:0Dclassifier_graph_1/sequential_1/output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2/
-classifier_graph_1/sequential_1/output/MatMul?
=classifier_graph_1/sequential_1/output/BiasAdd/ReadVariableOpReadVariableOpFclassifier_graph_1_sequential_1_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02?
=classifier_graph_1/sequential_1/output/BiasAdd/ReadVariableOp?
.classifier_graph_1/sequential_1/output/BiasAddBiasAdd7classifier_graph_1/sequential_1/output/MatMul:product:0Eclassifier_graph_1/sequential_1/output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????20
.classifier_graph_1/sequential_1/output/BiasAdd?
.classifier_graph_1/sequential_1/output/SoftmaxSoftmax7classifier_graph_1/sequential_1/output/BiasAdd:output:0*
T0*'
_output_shapes
:?????????20
.classifier_graph_1/sequential_1/output/Softmax?
IdentityIdentity8classifier_graph_1/sequential_1/output/Softmax:softmax:0?^classifier_graph_1/sequential_1/layer-1/BiasAdd/ReadVariableOp>^classifier_graph_1/sequential_1/layer-1/MatMul/ReadVariableOp>^classifier_graph_1/sequential_1/output/BiasAdd/ReadVariableOp=^classifier_graph_1/sequential_1/output/MatMul/ReadVariableOpB^classifier_graph_1/sequential_1/project_1/matmul_1/ReadVariableOpJ^classifier_graph_1/sequential_1/project_1/matrix_transpose/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????:::::2?
>classifier_graph_1/sequential_1/layer-1/BiasAdd/ReadVariableOp>classifier_graph_1/sequential_1/layer-1/BiasAdd/ReadVariableOp2~
=classifier_graph_1/sequential_1/layer-1/MatMul/ReadVariableOp=classifier_graph_1/sequential_1/layer-1/MatMul/ReadVariableOp2~
=classifier_graph_1/sequential_1/output/BiasAdd/ReadVariableOp=classifier_graph_1/sequential_1/output/BiasAdd/ReadVariableOp2|
<classifier_graph_1/sequential_1/output/MatMul/ReadVariableOp<classifier_graph_1/sequential_1/output/MatMul/ReadVariableOp2?
Aclassifier_graph_1/sequential_1/project_1/matmul_1/ReadVariableOpAclassifier_graph_1/sequential_1/project_1/matmul_1/ReadVariableOp2?
Iclassifier_graph_1/sequential_1/project_1/matrix_transpose/ReadVariableOpIclassifier_graph_1/sequential_1/project_1/matrix_transpose/ReadVariableOp:& "
 
_user_specified_nameinputs
?
?
2__inference_classifier_graph_1_layer_call_fn_81226
input_1"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1statefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5*
Tin

2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????**
config_proto

CPU

GPU 2J 8*V
fQRO
M__inference_classifier_graph_1_layer_call_and_return_conditional_losses_812082
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????:::::22
StatefulPartitionedCallStatefulPartitionedCall:' #
!
_user_specified_name	input_1
?
?
G__inference_sequential_1_layer_call_and_return_conditional_losses_81146

inputs,
(project_1_statefulpartitionedcall_args_1*
&layer_1_statefulpartitionedcall_args_1*
&layer_1_statefulpartitionedcall_args_2)
%output_statefulpartitionedcall_args_1)
%output_statefulpartitionedcall_args_2
identity??layer-1/StatefulPartitionedCall?output/StatefulPartitionedCall?!project_1/StatefulPartitionedCall?
!project_1/StatefulPartitionedCallStatefulPartitionedCallinputs(project_1_statefulpartitionedcall_args_1*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????**
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_project_1_layer_call_and_return_conditional_losses_810372#
!project_1/StatefulPartitionedCall?
layer-1/StatefulPartitionedCallStatefulPartitionedCall*project_1/StatefulPartitionedCall:output:0&layer_1_statefulpartitionedcall_args_1&layer_1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????2**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_layer-1_layer_call_and_return_conditional_losses_810612!
layer-1/StatefulPartitionedCall?
output/StatefulPartitionedCallStatefulPartitionedCall(layer-1/StatefulPartitionedCall:output:0%output_statefulpartitionedcall_args_1%output_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????**
config_proto

CPU

GPU 2J 8*J
fERC
A__inference_output_layer_call_and_return_conditional_losses_810842 
output/StatefulPartitionedCall?
IdentityIdentity'output/StatefulPartitionedCall:output:0 ^layer-1/StatefulPartitionedCall^output/StatefulPartitionedCall"^project_1/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????:::::2B
layer-1/StatefulPartitionedCalllayer-1/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall2F
!project_1/StatefulPartitionedCall!project_1/StatefulPartitionedCall:& "
 
_user_specified_nameinputs
?	
?
A__inference_output_layer_call_and_return_conditional_losses_81084

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
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
?
?
2__inference_classifier_graph_1_layer_call_fn_81216
input_1"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1statefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5*
Tin

2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????**
config_proto

CPU

GPU 2J 8*V
fQRO
M__inference_classifier_graph_1_layer_call_and_return_conditional_losses_812082
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????:::::22
StatefulPartitionedCallStatefulPartitionedCall:' #
!
_user_specified_name	input_1
?
?
%__inference_model_layer_call_fn_81407

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5*
Tin

2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????**
config_proto

CPU

GPU 2J 8*I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_813062
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????:::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
?!
?
G__inference_sequential_1_layer_call_and_return_conditional_losses_81541

inputs6
2project_1_matrix_transpose_readvariableop_resource*
&layer_1_matmul_readvariableop_resource+
'layer_1_biasadd_readvariableop_resource)
%output_matmul_readvariableop_resource*
&output_biasadd_readvariableop_resource
identity??layer-1/BiasAdd/ReadVariableOp?layer-1/MatMul/ReadVariableOp?output/BiasAdd/ReadVariableOp?output/MatMul/ReadVariableOp?!project_1/matmul_1/ReadVariableOp?)project_1/matrix_transpose/ReadVariableOp?
)project_1/matrix_transpose/ReadVariableOpReadVariableOp2project_1_matrix_transpose_readvariableop_resource*
_output_shapes

:*
dtype02+
)project_1/matrix_transpose/ReadVariableOp?
)project_1/matrix_transpose/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2+
)project_1/matrix_transpose/transpose/perm?
$project_1/matrix_transpose/transpose	Transpose1project_1/matrix_transpose/ReadVariableOp:value:02project_1/matrix_transpose/transpose/perm:output:0*
T0*
_output_shapes

:2&
$project_1/matrix_transpose/transpose?
project_1/matmulMatMulinputs(project_1/matrix_transpose/transpose:y:0*
T0*'
_output_shapes
:?????????2
project_1/matmul?
!project_1/matmul_1/ReadVariableOpReadVariableOp2project_1_matrix_transpose_readvariableop_resource*^project_1/matrix_transpose/ReadVariableOp*
_output_shapes

:*
dtype02#
!project_1/matmul_1/ReadVariableOp?
project_1/matmul_1MatMulproject_1/matmul:product:0)project_1/matmul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
project_1/matmul_1}
project_1/subSubinputsproject_1/matmul_1:product:0*
T0*'
_output_shapes
:?????????2
project_1/sub?
layer-1/MatMul/ReadVariableOpReadVariableOp&layer_1_matmul_readvariableop_resource*
_output_shapes

:2*
dtype02
layer-1/MatMul/ReadVariableOp?
layer-1/MatMulMatMulproject_1/sub:z:0%layer-1/MatMul/ReadVariableOp:value:0*
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
IdentityIdentityoutput/Softmax:softmax:0^layer-1/BiasAdd/ReadVariableOp^layer-1/MatMul/ReadVariableOp^output/BiasAdd/ReadVariableOp^output/MatMul/ReadVariableOp"^project_1/matmul_1/ReadVariableOp*^project_1/matrix_transpose/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????:::::2@
layer-1/BiasAdd/ReadVariableOplayer-1/BiasAdd/ReadVariableOp2>
layer-1/MatMul/ReadVariableOplayer-1/MatMul/ReadVariableOp2>
output/BiasAdd/ReadVariableOpoutput/BiasAdd/ReadVariableOp2<
output/MatMul/ReadVariableOpoutput/MatMul/ReadVariableOp2F
!project_1/matmul_1/ReadVariableOp!project_1/matmul_1/ReadVariableOp2V
)project_1/matrix_transpose/ReadVariableOp)project_1/matrix_transpose/ReadVariableOp:& "
 
_user_specified_nameinputs
?
?
D__inference_project_1_layer_call_and_return_conditional_losses_81037
x,
(matrix_transpose_readvariableop_resource
identity??matmul_1/ReadVariableOp?matrix_transpose/ReadVariableOp?
matrix_transpose/ReadVariableOpReadVariableOp(matrix_transpose_readvariableop_resource*
_output_shapes

:*
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

:2
matrix_transpose/transposeo
matmulMatMulxmatrix_transpose/transpose:y:0*
T0*'
_output_shapes
:?????????2
matmul?
matmul_1/ReadVariableOpReadVariableOp(matrix_transpose_readvariableop_resource ^matrix_transpose/ReadVariableOp*
_output_shapes

:*
dtype02
matmul_1/ReadVariableOp?
matmul_1MatMulmatmul:product:0matmul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2

matmul_1Z
subSubxmatmul_1:product:0*
T0*'
_output_shapes
:?????????2
sub?
IdentityIdentitysub:z:0^matmul_1/ReadVariableOp ^matrix_transpose/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:22
matmul_1/ReadVariableOpmatmul_1/ReadVariableOp2B
matrix_transpose/ReadVariableOpmatrix_transpose/ReadVariableOp:! 

_user_specified_namex
?
?
&__inference_output_layer_call_fn_81597

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????**
config_proto

CPU

GPU 2J 8*J
fERC
A__inference_output_layer_call_and_return_conditional_losses_810842
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????2::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
?
?
#__inference_signature_wrapper_81345
input_1"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1statefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5*
Tin

2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????**
config_proto

CPU

GPU 2J 8*)
f$R"
 __inference__wrapped_model_810242
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????:::::22
StatefulPartitionedCallStatefulPartitionedCall:' #
!
_user_specified_name	input_1
?)
?
M__inference_classifier_graph_1_layer_call_and_return_conditional_losses_81443
xC
?sequential_1_project_1_matrix_transpose_readvariableop_resource7
3sequential_1_layer_1_matmul_readvariableop_resource8
4sequential_1_layer_1_biasadd_readvariableop_resource6
2sequential_1_output_matmul_readvariableop_resource7
3sequential_1_output_biasadd_readvariableop_resource
identity??+sequential_1/layer-1/BiasAdd/ReadVariableOp?*sequential_1/layer-1/MatMul/ReadVariableOp?*sequential_1/output/BiasAdd/ReadVariableOp?)sequential_1/output/MatMul/ReadVariableOp?.sequential_1/project_1/matmul_1/ReadVariableOp?6sequential_1/project_1/matrix_transpose/ReadVariableOp?
6sequential_1/project_1/matrix_transpose/ReadVariableOpReadVariableOp?sequential_1_project_1_matrix_transpose_readvariableop_resource*
_output_shapes

:*
dtype028
6sequential_1/project_1/matrix_transpose/ReadVariableOp?
6sequential_1/project_1/matrix_transpose/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       28
6sequential_1/project_1/matrix_transpose/transpose/perm?
1sequential_1/project_1/matrix_transpose/transpose	Transpose>sequential_1/project_1/matrix_transpose/ReadVariableOp:value:0?sequential_1/project_1/matrix_transpose/transpose/perm:output:0*
T0*
_output_shapes

:23
1sequential_1/project_1/matrix_transpose/transpose?
sequential_1/project_1/matmulMatMulx5sequential_1/project_1/matrix_transpose/transpose:y:0*
T0*'
_output_shapes
:?????????2
sequential_1/project_1/matmul?
.sequential_1/project_1/matmul_1/ReadVariableOpReadVariableOp?sequential_1_project_1_matrix_transpose_readvariableop_resource7^sequential_1/project_1/matrix_transpose/ReadVariableOp*
_output_shapes

:*
dtype020
.sequential_1/project_1/matmul_1/ReadVariableOp?
sequential_1/project_1/matmul_1MatMul'sequential_1/project_1/matmul:product:06sequential_1/project_1/matmul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2!
sequential_1/project_1/matmul_1?
sequential_1/project_1/subSubx)sequential_1/project_1/matmul_1:product:0*
T0*'
_output_shapes
:?????????2
sequential_1/project_1/sub?
*sequential_1/layer-1/MatMul/ReadVariableOpReadVariableOp3sequential_1_layer_1_matmul_readvariableop_resource*
_output_shapes

:2*
dtype02,
*sequential_1/layer-1/MatMul/ReadVariableOp?
sequential_1/layer-1/MatMulMatMulsequential_1/project_1/sub:z:02sequential_1/layer-1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
sequential_1/layer-1/MatMul?
+sequential_1/layer-1/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_layer_1_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02-
+sequential_1/layer-1/BiasAdd/ReadVariableOp?
sequential_1/layer-1/BiasAddBiasAdd%sequential_1/layer-1/MatMul:product:03sequential_1/layer-1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
sequential_1/layer-1/BiasAdd?
sequential_1/layer-1/ReluRelu%sequential_1/layer-1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????22
sequential_1/layer-1/Relu?
)sequential_1/output/MatMul/ReadVariableOpReadVariableOp2sequential_1_output_matmul_readvariableop_resource*
_output_shapes

:2*
dtype02+
)sequential_1/output/MatMul/ReadVariableOp?
sequential_1/output/MatMulMatMul'sequential_1/layer-1/Relu:activations:01sequential_1/output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential_1/output/MatMul?
*sequential_1/output/BiasAdd/ReadVariableOpReadVariableOp3sequential_1_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02,
*sequential_1/output/BiasAdd/ReadVariableOp?
sequential_1/output/BiasAddBiasAdd$sequential_1/output/MatMul:product:02sequential_1/output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential_1/output/BiasAdd?
sequential_1/output/SoftmaxSoftmax$sequential_1/output/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
sequential_1/output/Softmax?
IdentityIdentity%sequential_1/output/Softmax:softmax:0,^sequential_1/layer-1/BiasAdd/ReadVariableOp+^sequential_1/layer-1/MatMul/ReadVariableOp+^sequential_1/output/BiasAdd/ReadVariableOp*^sequential_1/output/MatMul/ReadVariableOp/^sequential_1/project_1/matmul_1/ReadVariableOp7^sequential_1/project_1/matrix_transpose/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????:::::2Z
+sequential_1/layer-1/BiasAdd/ReadVariableOp+sequential_1/layer-1/BiasAdd/ReadVariableOp2X
*sequential_1/layer-1/MatMul/ReadVariableOp*sequential_1/layer-1/MatMul/ReadVariableOp2X
*sequential_1/output/BiasAdd/ReadVariableOp*sequential_1/output/BiasAdd/ReadVariableOp2V
)sequential_1/output/MatMul/ReadVariableOp)sequential_1/output/MatMul/ReadVariableOp2`
.sequential_1/project_1/matmul_1/ReadVariableOp.sequential_1/project_1/matmul_1/ReadVariableOp2p
6sequential_1/project_1/matrix_transpose/ReadVariableOp6sequential_1/project_1/matrix_transpose/ReadVariableOp:! 

_user_specified_namex
?

?
M__inference_classifier_graph_1_layer_call_and_return_conditional_losses_81185
input_1/
+sequential_1_statefulpartitionedcall_args_1/
+sequential_1_statefulpartitionedcall_args_2/
+sequential_1_statefulpartitionedcall_args_3/
+sequential_1_statefulpartitionedcall_args_4/
+sequential_1_statefulpartitionedcall_args_5
identity??$sequential_1/StatefulPartitionedCall?
$sequential_1/StatefulPartitionedCallStatefulPartitionedCallinput_1+sequential_1_statefulpartitionedcall_args_1+sequential_1_statefulpartitionedcall_args_2+sequential_1_statefulpartitionedcall_args_3+sequential_1_statefulpartitionedcall_args_4+sequential_1_statefulpartitionedcall_args_5*
Tin

2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????**
config_proto

CPU

GPU 2J 8*P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_811242&
$sequential_1/StatefulPartitionedCall?
IdentityIdentity-sequential_1/StatefulPartitionedCall:output:0%^sequential_1/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????:::::2L
$sequential_1/StatefulPartitionedCall$sequential_1/StatefulPartitionedCall:' #
!
_user_specified_name	input_1
?
?
%__inference_model_layer_call_fn_81334
input_1"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1statefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5*
Tin

2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????**
config_proto

CPU

GPU 2J 8*I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_813262
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????:::::22
StatefulPartitionedCallStatefulPartitionedCall:' #
!
_user_specified_name	input_1
?
?
@__inference_model_layer_call_and_return_conditional_losses_81326

inputs5
1classifier_graph_1_statefulpartitionedcall_args_15
1classifier_graph_1_statefulpartitionedcall_args_25
1classifier_graph_1_statefulpartitionedcall_args_35
1classifier_graph_1_statefulpartitionedcall_args_45
1classifier_graph_1_statefulpartitionedcall_args_5
identity??*classifier_graph_1/StatefulPartitionedCall?
*classifier_graph_1/StatefulPartitionedCallStatefulPartitionedCallinputs1classifier_graph_1_statefulpartitionedcall_args_11classifier_graph_1_statefulpartitionedcall_args_21classifier_graph_1_statefulpartitionedcall_args_31classifier_graph_1_statefulpartitionedcall_args_41classifier_graph_1_statefulpartitionedcall_args_5*
Tin

2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????**
config_proto

CPU

GPU 2J 8*V
fQRO
M__inference_classifier_graph_1_layer_call_and_return_conditional_losses_812082,
*classifier_graph_1/StatefulPartitionedCall?
IdentityIdentity3classifier_graph_1/StatefulPartitionedCall:output:0+^classifier_graph_1/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????:::::2X
*classifier_graph_1/StatefulPartitionedCall*classifier_graph_1/StatefulPartitionedCall:& "
 
_user_specified_nameinputs
?
?
@__inference_model_layer_call_and_return_conditional_losses_81283
input_15
1classifier_graph_1_statefulpartitionedcall_args_15
1classifier_graph_1_statefulpartitionedcall_args_25
1classifier_graph_1_statefulpartitionedcall_args_35
1classifier_graph_1_statefulpartitionedcall_args_45
1classifier_graph_1_statefulpartitionedcall_args_5
identity??*classifier_graph_1/StatefulPartitionedCall?
*classifier_graph_1/StatefulPartitionedCallStatefulPartitionedCallinput_11classifier_graph_1_statefulpartitionedcall_args_11classifier_graph_1_statefulpartitionedcall_args_21classifier_graph_1_statefulpartitionedcall_args_31classifier_graph_1_statefulpartitionedcall_args_41classifier_graph_1_statefulpartitionedcall_args_5*
Tin

2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????**
config_proto

CPU

GPU 2J 8*V
fQRO
M__inference_classifier_graph_1_layer_call_and_return_conditional_losses_812082,
*classifier_graph_1/StatefulPartitionedCall?
IdentityIdentity3classifier_graph_1/StatefulPartitionedCall:output:0+^classifier_graph_1/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????:::::2X
*classifier_graph_1/StatefulPartitionedCall*classifier_graph_1/StatefulPartitionedCall:' #
!
_user_specified_name	input_1
?
?
@__inference_model_layer_call_and_return_conditional_losses_81306

inputs5
1classifier_graph_1_statefulpartitionedcall_args_15
1classifier_graph_1_statefulpartitionedcall_args_25
1classifier_graph_1_statefulpartitionedcall_args_35
1classifier_graph_1_statefulpartitionedcall_args_45
1classifier_graph_1_statefulpartitionedcall_args_5
identity??*classifier_graph_1/StatefulPartitionedCall?
*classifier_graph_1/StatefulPartitionedCallStatefulPartitionedCallinputs1classifier_graph_1_statefulpartitionedcall_args_11classifier_graph_1_statefulpartitionedcall_args_21classifier_graph_1_statefulpartitionedcall_args_31classifier_graph_1_statefulpartitionedcall_args_41classifier_graph_1_statefulpartitionedcall_args_5*
Tin

2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????**
config_proto

CPU

GPU 2J 8*V
fQRO
M__inference_classifier_graph_1_layer_call_and_return_conditional_losses_812082,
*classifier_graph_1/StatefulPartitionedCall?
IdentityIdentity3classifier_graph_1/StatefulPartitionedCall:output:0+^classifier_graph_1/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????:::::2X
*classifier_graph_1/StatefulPartitionedCall*classifier_graph_1/StatefulPartitionedCall:& "
 
_user_specified_nameinputs
?
?
2__inference_classifier_graph_1_layer_call_fn_81479
x"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallxstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5*
Tin

2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????**
config_proto

CPU

GPU 2J 8*V
fQRO
M__inference_classifier_graph_1_layer_call_and_return_conditional_losses_812082
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????:::::22
StatefulPartitionedCallStatefulPartitionedCall:! 

_user_specified_namex
?
?
2__inference_classifier_graph_1_layer_call_fn_81489
x"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallxstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5*
Tin

2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????**
config_proto

CPU

GPU 2J 8*V
fQRO
M__inference_classifier_graph_1_layer_call_and_return_conditional_losses_812082
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????:::::22
StatefulPartitionedCallStatefulPartitionedCall:! 

_user_specified_namex
?
?
,__inference_sequential_1_layer_call_fn_81551

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5*
Tin

2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????**
config_proto

CPU

GPU 2J 8*P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_811242
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????:::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
?
?
G__inference_sequential_1_layer_call_and_return_conditional_losses_81109
input_1,
(project_1_statefulpartitionedcall_args_1*
&layer_1_statefulpartitionedcall_args_1*
&layer_1_statefulpartitionedcall_args_2)
%output_statefulpartitionedcall_args_1)
%output_statefulpartitionedcall_args_2
identity??layer-1/StatefulPartitionedCall?output/StatefulPartitionedCall?!project_1/StatefulPartitionedCall?
!project_1/StatefulPartitionedCallStatefulPartitionedCallinput_1(project_1_statefulpartitionedcall_args_1*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????**
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_project_1_layer_call_and_return_conditional_losses_810372#
!project_1/StatefulPartitionedCall?
layer-1/StatefulPartitionedCallStatefulPartitionedCall*project_1/StatefulPartitionedCall:output:0&layer_1_statefulpartitionedcall_args_1&layer_1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????2**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_layer-1_layer_call_and_return_conditional_losses_810612!
layer-1/StatefulPartitionedCall?
output/StatefulPartitionedCallStatefulPartitionedCall(layer-1/StatefulPartitionedCall:output:0%output_statefulpartitionedcall_args_1%output_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????**
config_proto

CPU

GPU 2J 8*J
fERC
A__inference_output_layer_call_and_return_conditional_losses_810842 
output/StatefulPartitionedCall?
IdentityIdentity'output/StatefulPartitionedCall:output:0 ^layer-1/StatefulPartitionedCall^output/StatefulPartitionedCall"^project_1/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????:::::2B
layer-1/StatefulPartitionedCalllayer-1/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall2F
!project_1/StatefulPartitionedCall!project_1/StatefulPartitionedCall:' #
!
_user_specified_name	input_1
?
?
G__inference_sequential_1_layer_call_and_return_conditional_losses_81124

inputs,
(project_1_statefulpartitionedcall_args_1*
&layer_1_statefulpartitionedcall_args_1*
&layer_1_statefulpartitionedcall_args_2)
%output_statefulpartitionedcall_args_1)
%output_statefulpartitionedcall_args_2
identity??layer-1/StatefulPartitionedCall?output/StatefulPartitionedCall?!project_1/StatefulPartitionedCall?
!project_1/StatefulPartitionedCallStatefulPartitionedCallinputs(project_1_statefulpartitionedcall_args_1*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????**
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_project_1_layer_call_and_return_conditional_losses_810372#
!project_1/StatefulPartitionedCall?
layer-1/StatefulPartitionedCallStatefulPartitionedCall*project_1/StatefulPartitionedCall:output:0&layer_1_statefulpartitionedcall_args_1&layer_1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????2**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_layer-1_layer_call_and_return_conditional_losses_810612!
layer-1/StatefulPartitionedCall?
output/StatefulPartitionedCallStatefulPartitionedCall(layer-1/StatefulPartitionedCall:output:0%output_statefulpartitionedcall_args_1%output_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????**
config_proto

CPU

GPU 2J 8*J
fERC
A__inference_output_layer_call_and_return_conditional_losses_810842 
output/StatefulPartitionedCall?
IdentityIdentity'output/StatefulPartitionedCall:output:0 ^layer-1/StatefulPartitionedCall^output/StatefulPartitionedCall"^project_1/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????:::::2B
layer-1/StatefulPartitionedCalllayer-1/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall2F
!project_1/StatefulPartitionedCall!project_1/StatefulPartitionedCall:& "
 
_user_specified_nameinputs"?L
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
serving_default_input_1:0?????????F
classifier_graph_10
StatefulPartitionedCall:0?????????tensorflow/serving/predict:̈
?
layer-0
layer_with_weights-0
layer-1
trainable_variables
regularization_losses
	variables
	keras_api

signatures
>__call__
*?&call_and_return_all_conditional_losses
@_default_save_signature"?
_tf_keras_model?{"class_name": "Model", "name": "model", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "is_graph_network": true, "keras_version": "2.2.4-tf", "backend": "tensorflow", "model_config": {"class_name": "Model"}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input_1", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": [null, 5], "config": {"batch_input_shape": [null, 5], "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}}
?

Layers
		model

trainable_variables
regularization_losses
	variables
	keras_api
A__call__
*B&call_and_return_all_conditional_losses"?
_tf_keras_model?{"class_name": "ClassifierGraph", "name": "classifier_graph_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "is_graph_network": false, "keras_version": "2.2.4-tf", "backend": "tensorflow", "model_config": {"class_name": "ClassifierGraph"}}
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
?
layer_regularization_losses
trainable_variables

layers
metrics
non_trainable_variables
regularization_losses
	variables
>__call__
@_default_save_signature
*?&call_and_return_all_conditional_losses
&?"call_and_return_conditional_losses"
_generic_user_object
,
Cserving_default"
signature_map
5
0
1
2"
trackable_list_wrapper
?
layer_with_weights-0
layer-0
layer-1
layer-2
trainable_variables
regularization_losses
	variables
	keras_api
D__call__
*E&call_and_return_all_conditional_losses"?
_tf_keras_sequential?{"class_name": "Sequential", "name": "sequential_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": [null, 5], "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}, "is_graph_network": false, "keras_version": "2.2.4-tf", "backend": "tensorflow", "model_config": {"class_name": "Sequential"}}
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
?
layer_regularization_losses

trainable_variables

layers
 metrics
!non_trainable_variables
regularization_losses
	variables
A__call__
*B&call_and_return_all_conditional_losses
&B"call_and_return_conditional_losses"
_generic_user_object
@:>22.classifier_graph_1/sequential_1/layer-1/kernel
::822,classifier_graph_1/sequential_1/layer-1/bias
?:=22-classifier_graph_1/sequential_1/output/kernel
9:72+classifier_graph_1/sequential_1/output/bias
:2Variable
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
?
w
"trainable_variables
#regularization_losses
$	variables
%	keras_api
F__call__
*G&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Project", "name": "project_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": [null, 5], "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?

kernel
bias
&trainable_variables
'regularization_losses
(	variables
)	keras_api
H__call__
*I&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "layer-1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": [null, 5], "config": {"name": "layer-1", "trainable": true, "batch_input_shape": [null, 5], "dtype": "float32", "units": 50, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 5}}}}
?

kernel
bias
*trainable_variables
+regularization_losses
,	variables
-	keras_api
J__call__
*K&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "output", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "output", "trainable": true, "dtype": "float32", "units": 2, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 50}}}}
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
?
.layer_regularization_losses
trainable_variables

/layers
0metrics
1non_trainable_variables
regularization_losses
	variables
D__call__
*E&call_and_return_all_conditional_losses
&E"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
<
0
1
2
	3"
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
'
0"
trackable_list_wrapper
?
2layer_regularization_losses
"trainable_variables

3layers
4metrics
5non_trainable_variables
#regularization_losses
$	variables
F__call__
*G&call_and_return_all_conditional_losses
&G"call_and_return_conditional_losses"
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
6layer_regularization_losses
&trainable_variables

7layers
8metrics
9non_trainable_variables
'regularization_losses
(	variables
H__call__
*I&call_and_return_all_conditional_losses
&I"call_and_return_conditional_losses"
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
:layer_regularization_losses
*trainable_variables

;layers
<metrics
=non_trainable_variables
+regularization_losses
,	variables
J__call__
*K&call_and_return_all_conditional_losses
&K"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
5
0
1
2"
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
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?2?
%__inference_model_layer_call_fn_81407
%__inference_model_layer_call_fn_81314
%__inference_model_layer_call_fn_81417
%__inference_model_layer_call_fn_81334?
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
@__inference_model_layer_call_and_return_conditional_losses_81371
@__inference_model_layer_call_and_return_conditional_losses_81283
@__inference_model_layer_call_and_return_conditional_losses_81397
@__inference_model_layer_call_and_return_conditional_losses_81293?
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
?2?
 __inference__wrapped_model_81024?
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
input_1?????????
?2?
2__inference_classifier_graph_1_layer_call_fn_81226
2__inference_classifier_graph_1_layer_call_fn_81489
2__inference_classifier_graph_1_layer_call_fn_81479
2__inference_classifier_graph_1_layer_call_fn_81216?
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
M__inference_classifier_graph_1_layer_call_and_return_conditional_losses_81185
M__inference_classifier_graph_1_layer_call_and_return_conditional_losses_81469
M__inference_classifier_graph_1_layer_call_and_return_conditional_losses_81195
M__inference_classifier_graph_1_layer_call_and_return_conditional_losses_81443?
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
2B0
#__inference_signature_wrapper_81345input_1
?2?
,__inference_sequential_1_layer_call_fn_81551
,__inference_sequential_1_layer_call_fn_81561
,__inference_sequential_1_layer_call_fn_81132
,__inference_sequential_1_layer_call_fn_81154?
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
G__inference_sequential_1_layer_call_and_return_conditional_losses_81109
G__inference_sequential_1_layer_call_and_return_conditional_losses_81541
G__inference_sequential_1_layer_call_and_return_conditional_losses_81515
G__inference_sequential_1_layer_call_and_return_conditional_losses_81097?
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
?2?
)__inference_project_1_layer_call_fn_81044?
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
??????????
?2?
D__inference_project_1_layer_call_and_return_conditional_losses_81037?
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
??????????
?2?
'__inference_layer-1_layer_call_fn_81579?
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
B__inference_layer-1_layer_call_and_return_conditional_losses_81572?
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
&__inference_output_layer_call_fn_81597?
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
A__inference_output_layer_call_and_return_conditional_losses_81590?
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
 __inference__wrapped_model_81024?0?-
&?#
!?
input_1?????????
? "G?D
B
classifier_graph_1,?)
classifier_graph_1??????????
M__inference_classifier_graph_1_layer_call_and_return_conditional_losses_81185h8?5
.?+
!?
input_1?????????
p 
p
? "%?"
?
0?????????
? ?
M__inference_classifier_graph_1_layer_call_and_return_conditional_losses_81195h8?5
.?+
!?
input_1?????????
p 
p 
? "%?"
?
0?????????
? ?
M__inference_classifier_graph_1_layer_call_and_return_conditional_losses_81443b2?/
(?%
?
x?????????
p 
p
? "%?"
?
0?????????
? ?
M__inference_classifier_graph_1_layer_call_and_return_conditional_losses_81469b2?/
(?%
?
x?????????
p 
p 
? "%?"
?
0?????????
? ?
2__inference_classifier_graph_1_layer_call_fn_81216[8?5
.?+
!?
input_1?????????
p 
p
? "???????????
2__inference_classifier_graph_1_layer_call_fn_81226[8?5
.?+
!?
input_1?????????
p 
p 
? "???????????
2__inference_classifier_graph_1_layer_call_fn_81479U2?/
(?%
?
x?????????
p 
p
? "???????????
2__inference_classifier_graph_1_layer_call_fn_81489U2?/
(?%
?
x?????????
p 
p 
? "???????????
B__inference_layer-1_layer_call_and_return_conditional_losses_81572\/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????2
? z
'__inference_layer-1_layer_call_fn_81579O/?,
%?"
 ?
inputs?????????
? "??????????2?
@__inference_model_layer_call_and_return_conditional_losses_81283h8?5
.?+
!?
input_1?????????
p

 
? "%?"
?
0?????????
? ?
@__inference_model_layer_call_and_return_conditional_losses_81293h8?5
.?+
!?
input_1?????????
p 

 
? "%?"
?
0?????????
? ?
@__inference_model_layer_call_and_return_conditional_losses_81371g7?4
-?*
 ?
inputs?????????
p

 
? "%?"
?
0?????????
? ?
@__inference_model_layer_call_and_return_conditional_losses_81397g7?4
-?*
 ?
inputs?????????
p 

 
? "%?"
?
0?????????
? ?
%__inference_model_layer_call_fn_81314[8?5
.?+
!?
input_1?????????
p

 
? "???????????
%__inference_model_layer_call_fn_81334[8?5
.?+
!?
input_1?????????
p 

 
? "???????????
%__inference_model_layer_call_fn_81407Z7?4
-?*
 ?
inputs?????????
p

 
? "???????????
%__inference_model_layer_call_fn_81417Z7?4
-?*
 ?
inputs?????????
p 

 
? "???????????
A__inference_output_layer_call_and_return_conditional_losses_81590\/?,
%?"
 ?
inputs?????????2
? "%?"
?
0?????????
? y
&__inference_output_layer_call_fn_81597O/?,
%?"
 ?
inputs?????????2
? "???????????
D__inference_project_1_layer_call_and_return_conditional_losses_81037V*?'
 ?
?
x?????????
? "%?"
?
0?????????
? v
)__inference_project_1_layer_call_fn_81044I*?'
 ?
?
x?????????
? "???????????
G__inference_sequential_1_layer_call_and_return_conditional_losses_81097h8?5
.?+
!?
input_1?????????
p

 
? "%?"
?
0?????????
? ?
G__inference_sequential_1_layer_call_and_return_conditional_losses_81109h8?5
.?+
!?
input_1?????????
p 

 
? "%?"
?
0?????????
? ?
G__inference_sequential_1_layer_call_and_return_conditional_losses_81515g7?4
-?*
 ?
inputs?????????
p

 
? "%?"
?
0?????????
? ?
G__inference_sequential_1_layer_call_and_return_conditional_losses_81541g7?4
-?*
 ?
inputs?????????
p 

 
? "%?"
?
0?????????
? ?
,__inference_sequential_1_layer_call_fn_81132[8?5
.?+
!?
input_1?????????
p

 
? "???????????
,__inference_sequential_1_layer_call_fn_81154[8?5
.?+
!?
input_1?????????
p 

 
? "???????????
,__inference_sequential_1_layer_call_fn_81551Z7?4
-?*
 ?
inputs?????????
p

 
? "???????????
,__inference_sequential_1_layer_call_fn_81561Z7?4
-?*
 ?
inputs?????????
p 

 
? "???????????
#__inference_signature_wrapper_81345?;?8
? 
1?.
,
input_1!?
input_1?????????"G?D
B
classifier_graph_1,?)
classifier_graph_1?????????