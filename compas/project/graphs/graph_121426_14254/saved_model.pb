??
??
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
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.3.02v2.3.0-rc2-23-gb36436b0878??
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
?
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?
value?B? B?
?
layer-0
layer_with_weights-0
layer-1
regularization_losses
trainable_variables
	variables
	keras_api

signatures
 
i

Layers
		model

regularization_losses
trainable_variables
	variables
	keras_api
 

0
1
2
3
#
0
1
2
3
4
?
regularization_losses
metrics
layer_metrics
trainable_variables
non_trainable_variables
layer_regularization_losses

layers
	variables
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
regularization_losses
trainable_variables
	variables
	keras_api
 

0
1
2
3
#
0
1
2
3
4
?

regularization_losses
metrics
 layer_metrics
trainable_variables
!non_trainable_variables
"layer_regularization_losses

#layers
	variables
TR
VARIABLE_VALUElayer-1/kernel0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUElayer-1/bias0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEoutput/kernel0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEoutput/bias0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUE
DB
VARIABLE_VALUEVariable&variables/4/.ATTRIBUTES/VARIABLE_VALUE
 
 

0
 

0
1
?
w
$_inbound_nodes
%_outbound_nodes
&regularization_losses
'trainable_variables
(	variables
)	keras_api
?
*_inbound_nodes

kernel
bias
+_outbound_nodes
,regularization_losses
-trainable_variables
.	variables
/	keras_api
|
0_inbound_nodes

kernel
bias
1regularization_losses
2trainable_variables
3	variables
4	keras_api
 

0
1
2
3
#
0
1
2
3
4
?
regularization_losses
5metrics
6layer_metrics
trainable_variables
7non_trainable_variables
8layer_regularization_losses

9layers
	variables
 
 

0
 

0
1
2
	3
 
 
 
 

0
?
&regularization_losses
:metrics
;layer_metrics
'trainable_variables
<non_trainable_variables
=layer_regularization_losses

>layers
(	variables
 
 
 

0
1

0
1
?
,regularization_losses
?metrics
@layer_metrics
-trainable_variables
Anon_trainable_variables
Blayer_regularization_losses

Clayers
.	variables
 
 

0
1

0
1
?
1regularization_losses
Dmetrics
Elayer_metrics
2trainable_variables
Fnon_trainable_variables
Glayer_regularization_losses

Hlayers
3	variables
 
 

0
 

0
1
2
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
 
z
serving_default_input_2Placeholder*'
_output_shapes
:?????????	*
dtype0*
shape:?????????	
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_2Variablelayer-1/kernellayer-1/biasoutput/kerneloutput/bias*
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
%__inference_signature_wrapper_1266315
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
 __inference__traced_save_1266803
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
#__inference__traced_restore_1266828ٿ
?
?
C__inference_output_layer_call_and_return_conditional_losses_1266756

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
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
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????2:::O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs
?
?
4__inference_classifier_graph_1_layer_call_fn_1266561
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
O__inference_classifier_graph_1_layer_call_and_return_conditional_losses_12661222
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
O__inference_classifier_graph_1_layer_call_and_return_conditional_losses_1266122
x
sequential_1_1266110
sequential_1_1266112
sequential_1_1266114
sequential_1_1266116
sequential_1_1266118
identity??$sequential_1/StatefulPartitionedCall?
$sequential_1/StatefulPartitionedCallStatefulPartitionedCallxsequential_1_1266110sequential_1_1266112sequential_1_1266114sequential_1_1266116sequential_1_1266118*
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
I__inference_sequential_1_layer_call_and_return_conditional_losses_12660302&
$sequential_1/StatefulPartitionedCall?
IdentityIdentity-sequential_1/StatefulPartitionedCall:output:0%^sequential_1/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????	:::::2L
$sequential_1/StatefulPartitionedCall$sequential_1/StatefulPartitionedCall:J F
'
_output_shapes
:?????????	

_user_specified_namex
?
?
.__inference_functional_3_layer_call_fn_1266298
input_2
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2	unknown_3*
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
I__inference_functional_3_layer_call_and_return_conditional_losses_12662852
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
_user_specified_name	input_2
?
?
.__inference_functional_3_layer_call_fn_1266382

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
I__inference_functional_3_layer_call_and_return_conditional_losses_12662552
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
?
~
)__inference_layer-1_layer_call_fn_1266745

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
D__inference_layer-1_layer_call_and_return_conditional_losses_12659172
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
.__inference_sequential_1_layer_call_fn_1266643

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
I__inference_sequential_1_layer_call_and_return_conditional_losses_12660302
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
?
O__inference_classifier_graph_1_layer_call_and_return_conditional_losses_1266179
xC
?sequential_1_project_1_matrix_transpose_readvariableop_resource7
3sequential_1_layer_1_matmul_readvariableop_resource8
4sequential_1_layer_1_biasadd_readvariableop_resource6
2sequential_1_output_matmul_readvariableop_resource7
3sequential_1_output_biasadd_readvariableop_resource
identity??
6sequential_1/project_1/matrix_transpose/ReadVariableOpReadVariableOp?sequential_1_project_1_matrix_transpose_readvariableop_resource*
_output_shapes

:	*
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

:	23
1sequential_1/project_1/matrix_transpose/transpose?
sequential_1/project_1/matmulMatMulx5sequential_1/project_1/matrix_transpose/transpose:y:0*
T0*'
_output_shapes
:?????????2
sequential_1/project_1/matmul?
.sequential_1/project_1/matmul_1/ReadVariableOpReadVariableOp?sequential_1_project_1_matrix_transpose_readvariableop_resource*
_output_shapes

:	*
dtype020
.sequential_1/project_1/matmul_1/ReadVariableOp?
sequential_1/project_1/matmul_1MatMul'sequential_1/project_1/matmul:product:06sequential_1/project_1/matmul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	2!
sequential_1/project_1/matmul_1?
sequential_1/project_1/subSubx)sequential_1/project_1/matmul_1:product:0*
T0*'
_output_shapes
:?????????	2
sequential_1/project_1/sub?
*sequential_1/layer-1/MatMul/ReadVariableOpReadVariableOp3sequential_1_layer_1_matmul_readvariableop_resource*
_output_shapes

:	2*
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
sequential_1/output/Softmaxy
IdentityIdentity%sequential_1/output/Softmax:softmax:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????	::::::J F
'
_output_shapes
:?????????	

_user_specified_namex
?
l
+__inference_project_1_layer_call_fn_1265899
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
F__inference_project_1_layer_call_and_return_conditional_losses_12658912
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
?
?
.__inference_sequential_1_layer_call_fn_1266710
project_1_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallproject_1_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3*
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
I__inference_sequential_1_layer_call_and_return_conditional_losses_12659982
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
_user_specified_nameproject_1_input
?
?
4__inference_classifier_graph_1_layer_call_fn_1266464
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
O__inference_classifier_graph_1_layer_call_and_return_conditional_losses_12661222
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
.__inference_functional_3_layer_call_fn_1266268
input_2
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2	unknown_3*
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
I__inference_functional_3_layer_call_and_return_conditional_losses_12662552
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
_user_specified_name	input_2
? 
?
O__inference_classifier_graph_1_layer_call_and_return_conditional_losses_1266449
input_1C
?sequential_1_project_1_matrix_transpose_readvariableop_resource7
3sequential_1_layer_1_matmul_readvariableop_resource8
4sequential_1_layer_1_biasadd_readvariableop_resource6
2sequential_1_output_matmul_readvariableop_resource7
3sequential_1_output_biasadd_readvariableop_resource
identity??
6sequential_1/project_1/matrix_transpose/ReadVariableOpReadVariableOp?sequential_1_project_1_matrix_transpose_readvariableop_resource*
_output_shapes

:	*
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

:	23
1sequential_1/project_1/matrix_transpose/transpose?
sequential_1/project_1/matmulMatMulinput_15sequential_1/project_1/matrix_transpose/transpose:y:0*
T0*'
_output_shapes
:?????????2
sequential_1/project_1/matmul?
.sequential_1/project_1/matmul_1/ReadVariableOpReadVariableOp?sequential_1_project_1_matrix_transpose_readvariableop_resource*
_output_shapes

:	*
dtype020
.sequential_1/project_1/matmul_1/ReadVariableOp?
sequential_1/project_1/matmul_1MatMul'sequential_1/project_1/matmul:product:06sequential_1/project_1/matmul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	2!
sequential_1/project_1/matmul_1?
sequential_1/project_1/subSubinput_1)sequential_1/project_1/matmul_1:product:0*
T0*'
_output_shapes
:?????????	2
sequential_1/project_1/sub?
*sequential_1/layer-1/MatMul/ReadVariableOpReadVariableOp3sequential_1_layer_1_matmul_readvariableop_resource*
_output_shapes

:	2*
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
sequential_1/output/Softmaxy
IdentityIdentity%sequential_1/output/Softmax:softmax:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????	::::::P L
'
_output_shapes
:?????????	
!
_user_specified_name	input_1
?
?
I__inference_sequential_1_layer_call_and_return_conditional_losses_1266613

inputs6
2project_1_matrix_transpose_readvariableop_resource*
&layer_1_matmul_readvariableop_resource+
'layer_1_biasadd_readvariableop_resource)
%output_matmul_readvariableop_resource*
&output_biasadd_readvariableop_resource
identity??
)project_1/matrix_transpose/ReadVariableOpReadVariableOp2project_1_matrix_transpose_readvariableop_resource*
_output_shapes

:	*
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

:	2&
$project_1/matrix_transpose/transpose?
project_1/matmulMatMulinputs(project_1/matrix_transpose/transpose:y:0*
T0*'
_output_shapes
:?????????2
project_1/matmul?
!project_1/matmul_1/ReadVariableOpReadVariableOp2project_1_matrix_transpose_readvariableop_resource*
_output_shapes

:	*
dtype02#
!project_1/matmul_1/ReadVariableOp?
project_1/matmul_1MatMulproject_1/matmul:product:0)project_1/matmul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	2
project_1/matmul_1}
project_1/subSubinputsproject_1/matmul_1:product:0*
T0*'
_output_shapes
:?????????	2
project_1/sub?
layer-1/MatMul/ReadVariableOpReadVariableOp&layer_1_matmul_readvariableop_resource*
_output_shapes

:	2*
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
output/Softmaxl
IdentityIdentityoutput/Softmax:softmax:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????	::::::O K
'
_output_shapes
:?????????	
 
_user_specified_nameinputs
?
?
 __inference__traced_save_1266803
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
Const?
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_cf262bf479f94a0fba08d9ea62603a76/part2	
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
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
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
4__inference_classifier_graph_1_layer_call_fn_1266546
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
O__inference_classifier_graph_1_layer_call_and_return_conditional_losses_12661222
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
.__inference_sequential_1_layer_call_fn_1266628

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
I__inference_sequential_1_layer_call_and_return_conditional_losses_12659982
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
.__inference_sequential_1_layer_call_fn_1266725
project_1_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallproject_1_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3*
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
I__inference_sequential_1_layer_call_and_return_conditional_losses_12660302
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
_user_specified_nameproject_1_input
?0
?
"__inference__wrapped_model_1265878
input_2c
_functional_3_classifier_graph_1_sequential_1_project_1_matrix_transpose_readvariableop_resourceW
Sfunctional_3_classifier_graph_1_sequential_1_layer_1_matmul_readvariableop_resourceX
Tfunctional_3_classifier_graph_1_sequential_1_layer_1_biasadd_readvariableop_resourceV
Rfunctional_3_classifier_graph_1_sequential_1_output_matmul_readvariableop_resourceW
Sfunctional_3_classifier_graph_1_sequential_1_output_biasadd_readvariableop_resource
identity??
Vfunctional_3/classifier_graph_1/sequential_1/project_1/matrix_transpose/ReadVariableOpReadVariableOp_functional_3_classifier_graph_1_sequential_1_project_1_matrix_transpose_readvariableop_resource*
_output_shapes

:	*
dtype02X
Vfunctional_3/classifier_graph_1/sequential_1/project_1/matrix_transpose/ReadVariableOp?
Vfunctional_3/classifier_graph_1/sequential_1/project_1/matrix_transpose/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2X
Vfunctional_3/classifier_graph_1/sequential_1/project_1/matrix_transpose/transpose/perm?
Qfunctional_3/classifier_graph_1/sequential_1/project_1/matrix_transpose/transpose	Transpose^functional_3/classifier_graph_1/sequential_1/project_1/matrix_transpose/ReadVariableOp:value:0_functional_3/classifier_graph_1/sequential_1/project_1/matrix_transpose/transpose/perm:output:0*
T0*
_output_shapes

:	2S
Qfunctional_3/classifier_graph_1/sequential_1/project_1/matrix_transpose/transpose?
=functional_3/classifier_graph_1/sequential_1/project_1/matmulMatMulinput_2Ufunctional_3/classifier_graph_1/sequential_1/project_1/matrix_transpose/transpose:y:0*
T0*'
_output_shapes
:?????????2?
=functional_3/classifier_graph_1/sequential_1/project_1/matmul?
Nfunctional_3/classifier_graph_1/sequential_1/project_1/matmul_1/ReadVariableOpReadVariableOp_functional_3_classifier_graph_1_sequential_1_project_1_matrix_transpose_readvariableop_resource*
_output_shapes

:	*
dtype02P
Nfunctional_3/classifier_graph_1/sequential_1/project_1/matmul_1/ReadVariableOp?
?functional_3/classifier_graph_1/sequential_1/project_1/matmul_1MatMulGfunctional_3/classifier_graph_1/sequential_1/project_1/matmul:product:0Vfunctional_3/classifier_graph_1/sequential_1/project_1/matmul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	2A
?functional_3/classifier_graph_1/sequential_1/project_1/matmul_1?
:functional_3/classifier_graph_1/sequential_1/project_1/subSubinput_2Ifunctional_3/classifier_graph_1/sequential_1/project_1/matmul_1:product:0*
T0*'
_output_shapes
:?????????	2<
:functional_3/classifier_graph_1/sequential_1/project_1/sub?
Jfunctional_3/classifier_graph_1/sequential_1/layer-1/MatMul/ReadVariableOpReadVariableOpSfunctional_3_classifier_graph_1_sequential_1_layer_1_matmul_readvariableop_resource*
_output_shapes

:	2*
dtype02L
Jfunctional_3/classifier_graph_1/sequential_1/layer-1/MatMul/ReadVariableOp?
;functional_3/classifier_graph_1/sequential_1/layer-1/MatMulMatMul>functional_3/classifier_graph_1/sequential_1/project_1/sub:z:0Rfunctional_3/classifier_graph_1/sequential_1/layer-1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22=
;functional_3/classifier_graph_1/sequential_1/layer-1/MatMul?
Kfunctional_3/classifier_graph_1/sequential_1/layer-1/BiasAdd/ReadVariableOpReadVariableOpTfunctional_3_classifier_graph_1_sequential_1_layer_1_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02M
Kfunctional_3/classifier_graph_1/sequential_1/layer-1/BiasAdd/ReadVariableOp?
<functional_3/classifier_graph_1/sequential_1/layer-1/BiasAddBiasAddEfunctional_3/classifier_graph_1/sequential_1/layer-1/MatMul:product:0Sfunctional_3/classifier_graph_1/sequential_1/layer-1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22>
<functional_3/classifier_graph_1/sequential_1/layer-1/BiasAdd?
9functional_3/classifier_graph_1/sequential_1/layer-1/ReluReluEfunctional_3/classifier_graph_1/sequential_1/layer-1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????22;
9functional_3/classifier_graph_1/sequential_1/layer-1/Relu?
Ifunctional_3/classifier_graph_1/sequential_1/output/MatMul/ReadVariableOpReadVariableOpRfunctional_3_classifier_graph_1_sequential_1_output_matmul_readvariableop_resource*
_output_shapes

:2*
dtype02K
Ifunctional_3/classifier_graph_1/sequential_1/output/MatMul/ReadVariableOp?
:functional_3/classifier_graph_1/sequential_1/output/MatMulMatMulGfunctional_3/classifier_graph_1/sequential_1/layer-1/Relu:activations:0Qfunctional_3/classifier_graph_1/sequential_1/output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2<
:functional_3/classifier_graph_1/sequential_1/output/MatMul?
Jfunctional_3/classifier_graph_1/sequential_1/output/BiasAdd/ReadVariableOpReadVariableOpSfunctional_3_classifier_graph_1_sequential_1_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02L
Jfunctional_3/classifier_graph_1/sequential_1/output/BiasAdd/ReadVariableOp?
;functional_3/classifier_graph_1/sequential_1/output/BiasAddBiasAddDfunctional_3/classifier_graph_1/sequential_1/output/MatMul:product:0Rfunctional_3/classifier_graph_1/sequential_1/output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2=
;functional_3/classifier_graph_1/sequential_1/output/BiasAdd?
;functional_3/classifier_graph_1/sequential_1/output/SoftmaxSoftmaxDfunctional_3/classifier_graph_1/sequential_1/output/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2=
;functional_3/classifier_graph_1/sequential_1/output/Softmax?
IdentityIdentityEfunctional_3/classifier_graph_1/sequential_1/output/Softmax:softmax:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????	::::::P L
'
_output_shapes
:?????????	
!
_user_specified_name	input_2
?
?
D__inference_layer-1_layer_call_and_return_conditional_losses_1265917

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
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
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????	:::O K
'
_output_shapes
:?????????	
 
_user_specified_nameinputs
?

?
F__inference_project_1_layer_call_and_return_conditional_losses_1265891
x,
(matrix_transpose_readvariableop_resource
identity??
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
sub[
IdentityIdentitysub:z:0*
T0*'
_output_shapes
:?????????	2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????	::J F
'
_output_shapes
:?????????	

_user_specified_namex
?

?
I__inference_functional_3_layer_call_and_return_conditional_losses_1266255

inputs
classifier_graph_1_1266243
classifier_graph_1_1266245
classifier_graph_1_1266247
classifier_graph_1_1266249
classifier_graph_1_1266251
identity??*classifier_graph_1/StatefulPartitionedCall?
*classifier_graph_1/StatefulPartitionedCallStatefulPartitionedCallinputsclassifier_graph_1_1266243classifier_graph_1_1266245classifier_graph_1_1266247classifier_graph_1_1266249classifier_graph_1_1266251*
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
O__inference_classifier_graph_1_layer_call_and_return_conditional_losses_12661792,
*classifier_graph_1/StatefulPartitionedCall?
IdentityIdentity3classifier_graph_1/StatefulPartitionedCall:output:0+^classifier_graph_1/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????	:::::2X
*classifier_graph_1/StatefulPartitionedCall*classifier_graph_1/StatefulPartitionedCall:O K
'
_output_shapes
:?????????	
 
_user_specified_nameinputs
?
}
(__inference_output_layer_call_fn_1266765

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
C__inference_output_layer_call_and_return_conditional_losses_12659442
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
?
?
4__inference_classifier_graph_1_layer_call_fn_1266479
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
O__inference_classifier_graph_1_layer_call_and_return_conditional_losses_12661222
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
.__inference_functional_3_layer_call_fn_1266397

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
I__inference_functional_3_layer_call_and_return_conditional_losses_12662852
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
?*
?
I__inference_functional_3_layer_call_and_return_conditional_losses_1266341

inputsV
Rclassifier_graph_1_sequential_1_project_1_matrix_transpose_readvariableop_resourceJ
Fclassifier_graph_1_sequential_1_layer_1_matmul_readvariableop_resourceK
Gclassifier_graph_1_sequential_1_layer_1_biasadd_readvariableop_resourceI
Eclassifier_graph_1_sequential_1_output_matmul_readvariableop_resourceJ
Fclassifier_graph_1_sequential_1_output_biasadd_readvariableop_resource
identity??
Iclassifier_graph_1/sequential_1/project_1/matrix_transpose/ReadVariableOpReadVariableOpRclassifier_graph_1_sequential_1_project_1_matrix_transpose_readvariableop_resource*
_output_shapes

:	*
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

:	2F
Dclassifier_graph_1/sequential_1/project_1/matrix_transpose/transpose?
0classifier_graph_1/sequential_1/project_1/matmulMatMulinputsHclassifier_graph_1/sequential_1/project_1/matrix_transpose/transpose:y:0*
T0*'
_output_shapes
:?????????22
0classifier_graph_1/sequential_1/project_1/matmul?
Aclassifier_graph_1/sequential_1/project_1/matmul_1/ReadVariableOpReadVariableOpRclassifier_graph_1_sequential_1_project_1_matrix_transpose_readvariableop_resource*
_output_shapes

:	*
dtype02C
Aclassifier_graph_1/sequential_1/project_1/matmul_1/ReadVariableOp?
2classifier_graph_1/sequential_1/project_1/matmul_1MatMul:classifier_graph_1/sequential_1/project_1/matmul:product:0Iclassifier_graph_1/sequential_1/project_1/matmul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	24
2classifier_graph_1/sequential_1/project_1/matmul_1?
-classifier_graph_1/sequential_1/project_1/subSubinputs<classifier_graph_1/sequential_1/project_1/matmul_1:product:0*
T0*'
_output_shapes
:?????????	2/
-classifier_graph_1/sequential_1/project_1/sub?
=classifier_graph_1/sequential_1/layer-1/MatMul/ReadVariableOpReadVariableOpFclassifier_graph_1_sequential_1_layer_1_matmul_readvariableop_resource*
_output_shapes

:	2*
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
.classifier_graph_1/sequential_1/output/Softmax?
IdentityIdentity8classifier_graph_1/sequential_1/output/Softmax:softmax:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????	::::::O K
'
_output_shapes
:?????????	
 
_user_specified_nameinputs
?
?
#__inference__traced_restore_1266828
file_prefix#
assignvariableop_layer_1_kernel#
assignvariableop_1_layer_1_bias$
 assignvariableop_2_output_kernel"
assignvariableop_3_output_bias
assignvariableop_4_variable

identity_6??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_2?AssignVariableOp_3?AssignVariableOp_4?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
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
?
O__inference_classifier_graph_1_layer_call_and_return_conditional_losses_1266505
xC
?sequential_1_project_1_matrix_transpose_readvariableop_resource7
3sequential_1_layer_1_matmul_readvariableop_resource8
4sequential_1_layer_1_biasadd_readvariableop_resource6
2sequential_1_output_matmul_readvariableop_resource7
3sequential_1_output_biasadd_readvariableop_resource
identity??
6sequential_1/project_1/matrix_transpose/ReadVariableOpReadVariableOp?sequential_1_project_1_matrix_transpose_readvariableop_resource*
_output_shapes

:	*
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

:	23
1sequential_1/project_1/matrix_transpose/transpose?
sequential_1/project_1/matmulMatMulx5sequential_1/project_1/matrix_transpose/transpose:y:0*
T0*'
_output_shapes
:?????????2
sequential_1/project_1/matmul?
.sequential_1/project_1/matmul_1/ReadVariableOpReadVariableOp?sequential_1_project_1_matrix_transpose_readvariableop_resource*
_output_shapes

:	*
dtype020
.sequential_1/project_1/matmul_1/ReadVariableOp?
sequential_1/project_1/matmul_1MatMul'sequential_1/project_1/matmul:product:06sequential_1/project_1/matmul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	2!
sequential_1/project_1/matmul_1?
sequential_1/project_1/subSubx)sequential_1/project_1/matmul_1:product:0*
T0*'
_output_shapes
:?????????	2
sequential_1/project_1/sub?
*sequential_1/layer-1/MatMul/ReadVariableOpReadVariableOp3sequential_1_layer_1_matmul_readvariableop_resource*
_output_shapes

:	2*
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
sequential_1/output/Softmaxy
IdentityIdentity%sequential_1/output/Softmax:softmax:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????	::::::J F
'
_output_shapes
:?????????	

_user_specified_namex
?
?
I__inference_sequential_1_layer_call_and_return_conditional_losses_1266587

inputs6
2project_1_matrix_transpose_readvariableop_resource*
&layer_1_matmul_readvariableop_resource+
'layer_1_biasadd_readvariableop_resource)
%output_matmul_readvariableop_resource*
&output_biasadd_readvariableop_resource
identity??
)project_1/matrix_transpose/ReadVariableOpReadVariableOp2project_1_matrix_transpose_readvariableop_resource*
_output_shapes

:	*
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

:	2&
$project_1/matrix_transpose/transpose?
project_1/matmulMatMulinputs(project_1/matrix_transpose/transpose:y:0*
T0*'
_output_shapes
:?????????2
project_1/matmul?
!project_1/matmul_1/ReadVariableOpReadVariableOp2project_1_matrix_transpose_readvariableop_resource*
_output_shapes

:	*
dtype02#
!project_1/matmul_1/ReadVariableOp?
project_1/matmul_1MatMulproject_1/matmul:product:0)project_1/matmul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	2
project_1/matmul_1}
project_1/subSubinputsproject_1/matmul_1:product:0*
T0*'
_output_shapes
:?????????	2
project_1/sub?
layer-1/MatMul/ReadVariableOpReadVariableOp&layer_1_matmul_readvariableop_resource*
_output_shapes

:	2*
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
output/Softmaxl
IdentityIdentityoutput/Softmax:softmax:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????	::::::O K
'
_output_shapes
:?????????	
 
_user_specified_nameinputs
?
?
I__inference_sequential_1_layer_call_and_return_conditional_losses_1266030

inputs
project_1_1266016
layer_1_1266019
layer_1_1266021
output_1266024
output_1266026
identity??layer-1/StatefulPartitionedCall?output/StatefulPartitionedCall?!project_1/StatefulPartitionedCall?
!project_1/StatefulPartitionedCallStatefulPartitionedCallinputsproject_1_1266016*
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
F__inference_project_1_layer_call_and_return_conditional_losses_12658912#
!project_1/StatefulPartitionedCall?
layer-1/StatefulPartitionedCallStatefulPartitionedCall*project_1/StatefulPartitionedCall:output:0layer_1_1266019layer_1_1266021*
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
D__inference_layer-1_layer_call_and_return_conditional_losses_12659172!
layer-1/StatefulPartitionedCall?
output/StatefulPartitionedCallStatefulPartitionedCall(layer-1/StatefulPartitionedCall:output:0output_1266024output_1266026*
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
C__inference_output_layer_call_and_return_conditional_losses_12659442 
output/StatefulPartitionedCall?
IdentityIdentity'output/StatefulPartitionedCall:output:0 ^layer-1/StatefulPartitionedCall^output/StatefulPartitionedCall"^project_1/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????	:::::2B
layer-1/StatefulPartitionedCalllayer-1/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall2F
!project_1/StatefulPartitionedCall!project_1/StatefulPartitionedCall:O K
'
_output_shapes
:?????????	
 
_user_specified_nameinputs
?
?
%__inference_signature_wrapper_1266315
input_2
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2	unknown_3*
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
"__inference__wrapped_model_12658782
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
_user_specified_name	input_2
?

?
I__inference_functional_3_layer_call_and_return_conditional_losses_1266222
input_2
classifier_graph_1_1266210
classifier_graph_1_1266212
classifier_graph_1_1266214
classifier_graph_1_1266216
classifier_graph_1_1266218
identity??*classifier_graph_1/StatefulPartitionedCall?
*classifier_graph_1/StatefulPartitionedCallStatefulPartitionedCallinput_2classifier_graph_1_1266210classifier_graph_1_1266212classifier_graph_1_1266214classifier_graph_1_1266216classifier_graph_1_1266218*
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
O__inference_classifier_graph_1_layer_call_and_return_conditional_losses_12661792,
*classifier_graph_1/StatefulPartitionedCall?
IdentityIdentity3classifier_graph_1/StatefulPartitionedCall:output:0+^classifier_graph_1/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????	:::::2X
*classifier_graph_1/StatefulPartitionedCall*classifier_graph_1/StatefulPartitionedCall:P L
'
_output_shapes
:?????????	
!
_user_specified_name	input_2
? 
?
O__inference_classifier_graph_1_layer_call_and_return_conditional_losses_1266423
input_1C
?sequential_1_project_1_matrix_transpose_readvariableop_resource7
3sequential_1_layer_1_matmul_readvariableop_resource8
4sequential_1_layer_1_biasadd_readvariableop_resource6
2sequential_1_output_matmul_readvariableop_resource7
3sequential_1_output_biasadd_readvariableop_resource
identity??
6sequential_1/project_1/matrix_transpose/ReadVariableOpReadVariableOp?sequential_1_project_1_matrix_transpose_readvariableop_resource*
_output_shapes

:	*
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

:	23
1sequential_1/project_1/matrix_transpose/transpose?
sequential_1/project_1/matmulMatMulinput_15sequential_1/project_1/matrix_transpose/transpose:y:0*
T0*'
_output_shapes
:?????????2
sequential_1/project_1/matmul?
.sequential_1/project_1/matmul_1/ReadVariableOpReadVariableOp?sequential_1_project_1_matrix_transpose_readvariableop_resource*
_output_shapes

:	*
dtype020
.sequential_1/project_1/matmul_1/ReadVariableOp?
sequential_1/project_1/matmul_1MatMul'sequential_1/project_1/matmul:product:06sequential_1/project_1/matmul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	2!
sequential_1/project_1/matmul_1?
sequential_1/project_1/subSubinput_1)sequential_1/project_1/matmul_1:product:0*
T0*'
_output_shapes
:?????????	2
sequential_1/project_1/sub?
*sequential_1/layer-1/MatMul/ReadVariableOpReadVariableOp3sequential_1_layer_1_matmul_readvariableop_resource*
_output_shapes

:	2*
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
sequential_1/output/Softmaxy
IdentityIdentity%sequential_1/output/Softmax:softmax:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????	::::::P L
'
_output_shapes
:?????????	
!
_user_specified_name	input_1
?
?
D__inference_layer-1_layer_call_and_return_conditional_losses_1266736

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
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
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????	:::O K
'
_output_shapes
:?????????	
 
_user_specified_nameinputs
?
?
I__inference_sequential_1_layer_call_and_return_conditional_losses_1265998

inputs
project_1_1265984
layer_1_1265987
layer_1_1265989
output_1265992
output_1265994
identity??layer-1/StatefulPartitionedCall?output/StatefulPartitionedCall?!project_1/StatefulPartitionedCall?
!project_1/StatefulPartitionedCallStatefulPartitionedCallinputsproject_1_1265984*
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
F__inference_project_1_layer_call_and_return_conditional_losses_12658912#
!project_1/StatefulPartitionedCall?
layer-1/StatefulPartitionedCallStatefulPartitionedCall*project_1/StatefulPartitionedCall:output:0layer_1_1265987layer_1_1265989*
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
D__inference_layer-1_layer_call_and_return_conditional_losses_12659172!
layer-1/StatefulPartitionedCall?
output/StatefulPartitionedCallStatefulPartitionedCall(layer-1/StatefulPartitionedCall:output:0output_1265992output_1265994*
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
C__inference_output_layer_call_and_return_conditional_losses_12659442 
output/StatefulPartitionedCall?
IdentityIdentity'output/StatefulPartitionedCall:output:0 ^layer-1/StatefulPartitionedCall^output/StatefulPartitionedCall"^project_1/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????	:::::2B
layer-1/StatefulPartitionedCalllayer-1/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall2F
!project_1/StatefulPartitionedCall!project_1/StatefulPartitionedCall:O K
'
_output_shapes
:?????????	
 
_user_specified_nameinputs
?
?
C__inference_output_layer_call_and_return_conditional_losses_1265944

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
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
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????2:::O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs
?*
?
I__inference_functional_3_layer_call_and_return_conditional_losses_1266367

inputsV
Rclassifier_graph_1_sequential_1_project_1_matrix_transpose_readvariableop_resourceJ
Fclassifier_graph_1_sequential_1_layer_1_matmul_readvariableop_resourceK
Gclassifier_graph_1_sequential_1_layer_1_biasadd_readvariableop_resourceI
Eclassifier_graph_1_sequential_1_output_matmul_readvariableop_resourceJ
Fclassifier_graph_1_sequential_1_output_biasadd_readvariableop_resource
identity??
Iclassifier_graph_1/sequential_1/project_1/matrix_transpose/ReadVariableOpReadVariableOpRclassifier_graph_1_sequential_1_project_1_matrix_transpose_readvariableop_resource*
_output_shapes

:	*
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

:	2F
Dclassifier_graph_1/sequential_1/project_1/matrix_transpose/transpose?
0classifier_graph_1/sequential_1/project_1/matmulMatMulinputsHclassifier_graph_1/sequential_1/project_1/matrix_transpose/transpose:y:0*
T0*'
_output_shapes
:?????????22
0classifier_graph_1/sequential_1/project_1/matmul?
Aclassifier_graph_1/sequential_1/project_1/matmul_1/ReadVariableOpReadVariableOpRclassifier_graph_1_sequential_1_project_1_matrix_transpose_readvariableop_resource*
_output_shapes

:	*
dtype02C
Aclassifier_graph_1/sequential_1/project_1/matmul_1/ReadVariableOp?
2classifier_graph_1/sequential_1/project_1/matmul_1MatMul:classifier_graph_1/sequential_1/project_1/matmul:product:0Iclassifier_graph_1/sequential_1/project_1/matmul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	24
2classifier_graph_1/sequential_1/project_1/matmul_1?
-classifier_graph_1/sequential_1/project_1/subSubinputs<classifier_graph_1/sequential_1/project_1/matmul_1:product:0*
T0*'
_output_shapes
:?????????	2/
-classifier_graph_1/sequential_1/project_1/sub?
=classifier_graph_1/sequential_1/layer-1/MatMul/ReadVariableOpReadVariableOpFclassifier_graph_1_sequential_1_layer_1_matmul_readvariableop_resource*
_output_shapes

:	2*
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
.classifier_graph_1/sequential_1/output/Softmax?
IdentityIdentity8classifier_graph_1/sequential_1/output/Softmax:softmax:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????	::::::O K
'
_output_shapes
:?????????	
 
_user_specified_nameinputs
?

?
I__inference_functional_3_layer_call_and_return_conditional_losses_1266237
input_2
classifier_graph_1_1266225
classifier_graph_1_1266227
classifier_graph_1_1266229
classifier_graph_1_1266231
classifier_graph_1_1266233
identity??*classifier_graph_1/StatefulPartitionedCall?
*classifier_graph_1/StatefulPartitionedCallStatefulPartitionedCallinput_2classifier_graph_1_1266225classifier_graph_1_1266227classifier_graph_1_1266229classifier_graph_1_1266231classifier_graph_1_1266233*
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
O__inference_classifier_graph_1_layer_call_and_return_conditional_losses_12661222,
*classifier_graph_1/StatefulPartitionedCall?
IdentityIdentity3classifier_graph_1/StatefulPartitionedCall:output:0+^classifier_graph_1/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????	:::::2X
*classifier_graph_1/StatefulPartitionedCall*classifier_graph_1/StatefulPartitionedCall:P L
'
_output_shapes
:?????????	
!
_user_specified_name	input_2
? 
?
O__inference_classifier_graph_1_layer_call_and_return_conditional_losses_1266531
xC
?sequential_1_project_1_matrix_transpose_readvariableop_resource7
3sequential_1_layer_1_matmul_readvariableop_resource8
4sequential_1_layer_1_biasadd_readvariableop_resource6
2sequential_1_output_matmul_readvariableop_resource7
3sequential_1_output_biasadd_readvariableop_resource
identity??
6sequential_1/project_1/matrix_transpose/ReadVariableOpReadVariableOp?sequential_1_project_1_matrix_transpose_readvariableop_resource*
_output_shapes

:	*
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

:	23
1sequential_1/project_1/matrix_transpose/transpose?
sequential_1/project_1/matmulMatMulx5sequential_1/project_1/matrix_transpose/transpose:y:0*
T0*'
_output_shapes
:?????????2
sequential_1/project_1/matmul?
.sequential_1/project_1/matmul_1/ReadVariableOpReadVariableOp?sequential_1_project_1_matrix_transpose_readvariableop_resource*
_output_shapes

:	*
dtype020
.sequential_1/project_1/matmul_1/ReadVariableOp?
sequential_1/project_1/matmul_1MatMul'sequential_1/project_1/matmul:product:06sequential_1/project_1/matmul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	2!
sequential_1/project_1/matmul_1?
sequential_1/project_1/subSubx)sequential_1/project_1/matmul_1:product:0*
T0*'
_output_shapes
:?????????	2
sequential_1/project_1/sub?
*sequential_1/layer-1/MatMul/ReadVariableOpReadVariableOp3sequential_1_layer_1_matmul_readvariableop_resource*
_output_shapes

:	2*
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
sequential_1/output/Softmaxy
IdentityIdentity%sequential_1/output/Softmax:softmax:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????	::::::J F
'
_output_shapes
:?????????	

_user_specified_namex
?
?
I__inference_sequential_1_layer_call_and_return_conditional_losses_1266669
project_1_input6
2project_1_matrix_transpose_readvariableop_resource*
&layer_1_matmul_readvariableop_resource+
'layer_1_biasadd_readvariableop_resource)
%output_matmul_readvariableop_resource*
&output_biasadd_readvariableop_resource
identity??
)project_1/matrix_transpose/ReadVariableOpReadVariableOp2project_1_matrix_transpose_readvariableop_resource*
_output_shapes

:	*
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

:	2&
$project_1/matrix_transpose/transpose?
project_1/matmulMatMulproject_1_input(project_1/matrix_transpose/transpose:y:0*
T0*'
_output_shapes
:?????????2
project_1/matmul?
!project_1/matmul_1/ReadVariableOpReadVariableOp2project_1_matrix_transpose_readvariableop_resource*
_output_shapes

:	*
dtype02#
!project_1/matmul_1/ReadVariableOp?
project_1/matmul_1MatMulproject_1/matmul:product:0)project_1/matmul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	2
project_1/matmul_1?
project_1/subSubproject_1_inputproject_1/matmul_1:product:0*
T0*'
_output_shapes
:?????????	2
project_1/sub?
layer-1/MatMul/ReadVariableOpReadVariableOp&layer_1_matmul_readvariableop_resource*
_output_shapes

:	2*
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
output/Softmaxl
IdentityIdentityoutput/Softmax:softmax:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????	::::::X T
'
_output_shapes
:?????????	
)
_user_specified_nameproject_1_input
?

?
I__inference_functional_3_layer_call_and_return_conditional_losses_1266285

inputs
classifier_graph_1_1266273
classifier_graph_1_1266275
classifier_graph_1_1266277
classifier_graph_1_1266279
classifier_graph_1_1266281
identity??*classifier_graph_1/StatefulPartitionedCall?
*classifier_graph_1/StatefulPartitionedCallStatefulPartitionedCallinputsclassifier_graph_1_1266273classifier_graph_1_1266275classifier_graph_1_1266277classifier_graph_1_1266279classifier_graph_1_1266281*
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
O__inference_classifier_graph_1_layer_call_and_return_conditional_losses_12661222,
*classifier_graph_1/StatefulPartitionedCall?
IdentityIdentity3classifier_graph_1/StatefulPartitionedCall:output:0+^classifier_graph_1/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????	:::::2X
*classifier_graph_1/StatefulPartitionedCall*classifier_graph_1/StatefulPartitionedCall:O K
'
_output_shapes
:?????????	
 
_user_specified_nameinputs
?
?
I__inference_sequential_1_layer_call_and_return_conditional_losses_1266695
project_1_input6
2project_1_matrix_transpose_readvariableop_resource*
&layer_1_matmul_readvariableop_resource+
'layer_1_biasadd_readvariableop_resource)
%output_matmul_readvariableop_resource*
&output_biasadd_readvariableop_resource
identity??
)project_1/matrix_transpose/ReadVariableOpReadVariableOp2project_1_matrix_transpose_readvariableop_resource*
_output_shapes

:	*
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

:	2&
$project_1/matrix_transpose/transpose?
project_1/matmulMatMulproject_1_input(project_1/matrix_transpose/transpose:y:0*
T0*'
_output_shapes
:?????????2
project_1/matmul?
!project_1/matmul_1/ReadVariableOpReadVariableOp2project_1_matrix_transpose_readvariableop_resource*
_output_shapes

:	*
dtype02#
!project_1/matmul_1/ReadVariableOp?
project_1/matmul_1MatMulproject_1/matmul:product:0)project_1/matmul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	2
project_1/matmul_1?
project_1/subSubproject_1_inputproject_1/matmul_1:product:0*
T0*'
_output_shapes
:?????????	2
project_1/sub?
layer-1/MatMul/ReadVariableOpReadVariableOp&layer_1_matmul_readvariableop_resource*
_output_shapes

:	2*
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
output/Softmaxl
IdentityIdentityoutput/Softmax:softmax:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????	::::::X T
'
_output_shapes
:?????????	
)
_user_specified_nameproject_1_input"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
;
input_20
serving_default_input_2:0?????????	F
classifier_graph_10
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
?	
layer-0
layer_with_weights-0
layer-1
regularization_losses
trainable_variables
	variables
	keras_api

signatures
I_default_save_signature
J__call__
*K&call_and_return_all_conditional_losses"?
_tf_keras_network?{"class_name": "Functional", "name": "functional_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "functional_3", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 9]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}, "name": "input_2", "inbound_nodes": []}, {"class_name": "ClassifierGraph", "config": {"layer was saved without config": true}, "name": "classifier_graph_1", "inbound_nodes": [[["input_2", 0, 0, {}]]]}], "input_layers": [["input_2", 0, 0]], "output_layers": [["classifier_graph_1", 0, 0]]}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 9]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional"}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input_2", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 9]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 9]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}}
?

Layers
		model

regularization_losses
trainable_variables
	variables
	keras_api
L__call__
*M&call_and_return_all_conditional_losses"?
_tf_keras_model?{"class_name": "ClassifierGraph", "name": "classifier_graph_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "is_graph_network": false, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "ClassifierGraph"}}
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
?
regularization_losses
metrics
layer_metrics
trainable_variables
non_trainable_variables
layer_regularization_losses

layers
	variables
J__call__
I_default_save_signature
*K&call_and_return_all_conditional_losses
&K"call_and_return_conditional_losses"
_generic_user_object
,
Nserving_default"
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
regularization_losses
trainable_variables
	variables
	keras_api
O__call__
*P&call_and_return_all_conditional_losses"?
_tf_keras_sequential?{"class_name": "Sequential", "name": "sequential_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 9]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "project_1_input"}}, {"class_name": "Project", "config": {"layer was saved without config": true}}, {"class_name": "Dense", "config": {"name": "layer-1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 9]}, "dtype": "float32", "units": 50, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "output", "trainable": true, "dtype": "float32", "units": 2, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 9]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 9]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential"}}
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
?

regularization_losses
metrics
 layer_metrics
trainable_variables
!non_trainable_variables
"layer_regularization_losses

#layers
	variables
L__call__
*M&call_and_return_all_conditional_losses
&M"call_and_return_conditional_losses"
_generic_user_object
 :	22layer-1/kernel
:22layer-1/bias
:22output/kernel
:2output/bias
:	2Variable
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
w
$_inbound_nodes
%_outbound_nodes
&regularization_losses
'trainable_variables
(	variables
)	keras_api
Q__call__
*R&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Project", "name": "project_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 9]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?	
*_inbound_nodes

kernel
bias
+_outbound_nodes
,regularization_losses
-trainable_variables
.	variables
/	keras_api
S__call__
*T&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "layer-1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 9]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "layer-1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 9]}, "dtype": "float32", "units": 50, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 9}}}, "build_input_shape": {"class_name": "TensorShape", "items": [250, 9]}}
?
0_inbound_nodes

kernel
bias
1regularization_losses
2trainable_variables
3	variables
4	keras_api
U__call__
*V&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "output", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "output", "trainable": true, "dtype": "float32", "units": 2, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 50}}}, "build_input_shape": {"class_name": "TensorShape", "items": [250, 50]}}
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
?
regularization_losses
5metrics
6layer_metrics
trainable_variables
7non_trainable_variables
8layer_regularization_losses

9layers
	variables
O__call__
*P&call_and_return_all_conditional_losses
&P"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
<
0
1
2
	3"
trackable_list_wrapper
 "
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
?
&regularization_losses
:metrics
;layer_metrics
'trainable_variables
<non_trainable_variables
=layer_regularization_losses

>layers
(	variables
Q__call__
*R&call_and_return_all_conditional_losses
&R"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
,regularization_losses
?metrics
@layer_metrics
-trainable_variables
Anon_trainable_variables
Blayer_regularization_losses

Clayers
.	variables
S__call__
*T&call_and_return_all_conditional_losses
&T"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
1regularization_losses
Dmetrics
Elayer_metrics
2trainable_variables
Fnon_trainable_variables
Glayer_regularization_losses

Hlayers
3	variables
U__call__
*V&call_and_return_all_conditional_losses
&V"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
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
?2?
"__inference__wrapped_model_1265878?
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
input_2?????????	
?2?
.__inference_functional_3_layer_call_fn_1266382
.__inference_functional_3_layer_call_fn_1266298
.__inference_functional_3_layer_call_fn_1266268
.__inference_functional_3_layer_call_fn_1266397?
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
I__inference_functional_3_layer_call_and_return_conditional_losses_1266367
I__inference_functional_3_layer_call_and_return_conditional_losses_1266341
I__inference_functional_3_layer_call_and_return_conditional_losses_1266237
I__inference_functional_3_layer_call_and_return_conditional_losses_1266222?
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
4__inference_classifier_graph_1_layer_call_fn_1266546
4__inference_classifier_graph_1_layer_call_fn_1266464
4__inference_classifier_graph_1_layer_call_fn_1266479
4__inference_classifier_graph_1_layer_call_fn_1266561?
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
?2?
O__inference_classifier_graph_1_layer_call_and_return_conditional_losses_1266449
O__inference_classifier_graph_1_layer_call_and_return_conditional_losses_1266531
O__inference_classifier_graph_1_layer_call_and_return_conditional_losses_1266423
O__inference_classifier_graph_1_layer_call_and_return_conditional_losses_1266505?
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
4B2
%__inference_signature_wrapper_1266315input_2
?2?
.__inference_sequential_1_layer_call_fn_1266710
.__inference_sequential_1_layer_call_fn_1266643
.__inference_sequential_1_layer_call_fn_1266725
.__inference_sequential_1_layer_call_fn_1266628?
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
I__inference_sequential_1_layer_call_and_return_conditional_losses_1266695
I__inference_sequential_1_layer_call_and_return_conditional_losses_1266587
I__inference_sequential_1_layer_call_and_return_conditional_losses_1266669
I__inference_sequential_1_layer_call_and_return_conditional_losses_1266613?
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
+__inference_project_1_layer_call_fn_1265899?
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
?2?
F__inference_project_1_layer_call_and_return_conditional_losses_1265891?
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
)__inference_layer-1_layer_call_fn_1266745?
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
D__inference_layer-1_layer_call_and_return_conditional_losses_1266736?
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
(__inference_output_layer_call_fn_1266765?
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
C__inference_output_layer_call_and_return_conditional_losses_1266756?
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
"__inference__wrapped_model_1265878?0?-
&?#
!?
input_2?????????	
? "G?D
B
classifier_graph_1,?)
classifier_graph_1??????????
O__inference_classifier_graph_1_layer_call_and_return_conditional_losses_1266423h8?5
.?+
!?
input_1?????????	
p 
p
? "%?"
?
0?????????
? ?
O__inference_classifier_graph_1_layer_call_and_return_conditional_losses_1266449h8?5
.?+
!?
input_1?????????	
p 
p 
? "%?"
?
0?????????
? ?
O__inference_classifier_graph_1_layer_call_and_return_conditional_losses_1266505b2?/
(?%
?
x?????????	
p 
p
? "%?"
?
0?????????
? ?
O__inference_classifier_graph_1_layer_call_and_return_conditional_losses_1266531b2?/
(?%
?
x?????????	
p 
p 
? "%?"
?
0?????????
? ?
4__inference_classifier_graph_1_layer_call_fn_1266464[8?5
.?+
!?
input_1?????????	
p 
p
? "???????????
4__inference_classifier_graph_1_layer_call_fn_1266479[8?5
.?+
!?
input_1?????????	
p 
p 
? "???????????
4__inference_classifier_graph_1_layer_call_fn_1266546U2?/
(?%
?
x?????????	
p 
p
? "???????????
4__inference_classifier_graph_1_layer_call_fn_1266561U2?/
(?%
?
x?????????	
p 
p 
? "???????????
I__inference_functional_3_layer_call_and_return_conditional_losses_1266222h8?5
.?+
!?
input_2?????????	
p

 
? "%?"
?
0?????????
? ?
I__inference_functional_3_layer_call_and_return_conditional_losses_1266237h8?5
.?+
!?
input_2?????????	
p 

 
? "%?"
?
0?????????
? ?
I__inference_functional_3_layer_call_and_return_conditional_losses_1266341g7?4
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
I__inference_functional_3_layer_call_and_return_conditional_losses_1266367g7?4
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
.__inference_functional_3_layer_call_fn_1266268[8?5
.?+
!?
input_2?????????	
p

 
? "???????????
.__inference_functional_3_layer_call_fn_1266298[8?5
.?+
!?
input_2?????????	
p 

 
? "???????????
.__inference_functional_3_layer_call_fn_1266382Z7?4
-?*
 ?
inputs?????????	
p

 
? "???????????
.__inference_functional_3_layer_call_fn_1266397Z7?4
-?*
 ?
inputs?????????	
p 

 
? "???????????
D__inference_layer-1_layer_call_and_return_conditional_losses_1266736\/?,
%?"
 ?
inputs?????????	
? "%?"
?
0?????????2
? |
)__inference_layer-1_layer_call_fn_1266745O/?,
%?"
 ?
inputs?????????	
? "??????????2?
C__inference_output_layer_call_and_return_conditional_losses_1266756\/?,
%?"
 ?
inputs?????????2
? "%?"
?
0?????????
? {
(__inference_output_layer_call_fn_1266765O/?,
%?"
 ?
inputs?????????2
? "???????????
F__inference_project_1_layer_call_and_return_conditional_losses_1265891V*?'
 ?
?
x?????????	
? "%?"
?
0?????????	
? x
+__inference_project_1_layer_call_fn_1265899I*?'
 ?
?
x?????????	
? "??????????	?
I__inference_sequential_1_layer_call_and_return_conditional_losses_1266587g7?4
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
I__inference_sequential_1_layer_call_and_return_conditional_losses_1266613g7?4
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
I__inference_sequential_1_layer_call_and_return_conditional_losses_1266669p@?=
6?3
)?&
project_1_input?????????	
p

 
? "%?"
?
0?????????
? ?
I__inference_sequential_1_layer_call_and_return_conditional_losses_1266695p@?=
6?3
)?&
project_1_input?????????	
p 

 
? "%?"
?
0?????????
? ?
.__inference_sequential_1_layer_call_fn_1266628Z7?4
-?*
 ?
inputs?????????	
p

 
? "???????????
.__inference_sequential_1_layer_call_fn_1266643Z7?4
-?*
 ?
inputs?????????	
p 

 
? "???????????
.__inference_sequential_1_layer_call_fn_1266710c@?=
6?3
)?&
project_1_input?????????	
p

 
? "???????????
.__inference_sequential_1_layer_call_fn_1266725c@?=
6?3
)?&
project_1_input?????????	
p 

 
? "???????????
%__inference_signature_wrapper_1266315?;?8
? 
1?.
,
input_2!?
input_2?????????	"G?D
B
classifier_graph_1,?)
classifier_graph_1?????????