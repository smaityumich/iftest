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
trainable_variables
	variables
regularization_losses
	keras_api

signatures
 
i

Layers
		model

trainable_variables
	variables
regularization_losses
	keras_api

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
 
?
layer_regularization_losses

layers
trainable_variables
	variables
non_trainable_variables
regularization_losses
metrics
layer_metrics
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
trainable_variables
	variables
regularization_losses
	keras_api

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
 
?
layer_regularization_losses

 layers

trainable_variables
	variables
!non_trainable_variables
regularization_losses
"metrics
#layer_metrics
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

0
1

0
 
 
?
w
$_inbound_nodes
%_outbound_nodes
&trainable_variables
'	variables
(regularization_losses
)	keras_api
?
*_inbound_nodes

kernel
bias
+_outbound_nodes
,trainable_variables
-	variables
.regularization_losses
/	keras_api
|
0_inbound_nodes

kernel
bias
1trainable_variables
2	variables
3regularization_losses
4	keras_api

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
 
?
5layer_regularization_losses

6layers
trainable_variables
	variables
7non_trainable_variables
regularization_losses
8metrics
9layer_metrics
 

0
1
2
	3

0
 
 
 
 
 

0
 
?
:layer_regularization_losses

;layers
&trainable_variables
'	variables
<non_trainable_variables
(regularization_losses
=metrics
>layer_metrics
 
 

0
1

0
1
 
?
?layer_regularization_losses

@layers
,trainable_variables
-	variables
Anon_trainable_variables
.regularization_losses
Bmetrics
Clayer_metrics
 

0
1

0
1
 
?
Dlayer_regularization_losses

Elayers
1trainable_variables
2	variables
Fnon_trainable_variables
3regularization_losses
Gmetrics
Hlayer_metrics
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
$__inference_signature_wrapper_632887
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
__inference__traced_save_633375
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
"__inference__traced_restore_633400??
?
?
F__inference_sequential_layer_call_and_return_conditional_losses_632602

inputs
project_632588
layer_1_632591
layer_1_632593
output_632596
output_632598
identity??layer-1/StatefulPartitionedCall?output/StatefulPartitionedCall?project/StatefulPartitionedCall?
project/StatefulPartitionedCallStatefulPartitionedCallinputsproject_632588*
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
C__inference_project_layer_call_and_return_conditional_losses_6324632!
project/StatefulPartitionedCall?
layer-1/StatefulPartitionedCallStatefulPartitionedCall(project/StatefulPartitionedCall:output:0layer_1_632591layer_1_632593*
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
C__inference_layer-1_layer_call_and_return_conditional_losses_6324892!
layer-1/StatefulPartitionedCall?
output/StatefulPartitionedCallStatefulPartitionedCall(layer-1/StatefulPartitionedCall:output:0output_632596output_632598*
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
B__inference_output_layer_call_and_return_conditional_losses_6325162 
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
?
?
F__inference_sequential_layer_call_and_return_conditional_losses_633159

inputs4
0project_matrix_transpose_readvariableop_resource*
&layer_1_matmul_readvariableop_resource+
'layer_1_biasadd_readvariableop_resource)
%output_matmul_readvariableop_resource*
&output_biasadd_readvariableop_resource
identity??
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
?
?
1__inference_classifier_graph_layer_call_fn_633036
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
L__inference_classifier_graph_layer_call_and_return_conditional_losses_6326942
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
1__inference_classifier_graph_layer_call_fn_633118
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
L__inference_classifier_graph_layer_call_and_return_conditional_losses_6326942
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
+__inference_sequential_layer_call_fn_633282
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
F__inference_sequential_layer_call_and_return_conditional_losses_6325702
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
?	
?
H__inference_functional_1_layer_call_and_return_conditional_losses_632827

inputs
classifier_graph_632815
classifier_graph_632817
classifier_graph_632819
classifier_graph_632821
classifier_graph_632823
identity??(classifier_graph/StatefulPartitionedCall?
(classifier_graph/StatefulPartitionedCallStatefulPartitionedCallinputsclassifier_graph_632815classifier_graph_632817classifier_graph_632819classifier_graph_632821classifier_graph_632823*
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
L__inference_classifier_graph_layer_call_and_return_conditional_losses_6327512*
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
1__inference_classifier_graph_layer_call_fn_633051
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
L__inference_classifier_graph_layer_call_and_return_conditional_losses_6326942
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
?
L__inference_classifier_graph_layer_call_and_return_conditional_losses_632751
x?
;sequential_project_matrix_transpose_readvariableop_resource5
1sequential_layer_1_matmul_readvariableop_resource6
2sequential_layer_1_biasadd_readvariableop_resource4
0sequential_output_matmul_readvariableop_resource5
1sequential_output_biasadd_readvariableop_resource
identity??
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
sequential/output/Softmaxw
IdentityIdentity#sequential/output/Softmax:softmax:0*
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
?
?
L__inference_classifier_graph_layer_call_and_return_conditional_losses_632694
x
sequential_632682
sequential_632684
sequential_632686
sequential_632688
sequential_632690
identity??"sequential/StatefulPartitionedCall?
"sequential/StatefulPartitionedCallStatefulPartitionedCallxsequential_632682sequential_632684sequential_632686sequential_632688sequential_632690*
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
F__inference_sequential_layer_call_and_return_conditional_losses_6326022$
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
?
?
L__inference_classifier_graph_layer_call_and_return_conditional_losses_632995
x?
;sequential_project_matrix_transpose_readvariableop_resource5
1sequential_layer_1_matmul_readvariableop_resource6
2sequential_layer_1_biasadd_readvariableop_resource4
0sequential_output_matmul_readvariableop_resource5
1sequential_output_biasadd_readvariableop_resource
identity??
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
sequential/output/Softmaxw
IdentityIdentity#sequential/output/Softmax:softmax:0*
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
?
?
1__inference_classifier_graph_layer_call_fn_633133
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
L__inference_classifier_graph_layer_call_and_return_conditional_losses_6326942
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
?
L__inference_classifier_graph_layer_call_and_return_conditional_losses_633103
input_1?
;sequential_project_matrix_transpose_readvariableop_resource5
1sequential_layer_1_matmul_readvariableop_resource6
2sequential_layer_1_biasadd_readvariableop_resource4
0sequential_output_matmul_readvariableop_resource5
1sequential_output_biasadd_readvariableop_resource
identity??
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
sequential/output/Softmaxw
IdentityIdentity#sequential/output/Softmax:softmax:0*
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
__inference__traced_save_633375
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
value3B1 B+_temp_317c0926663c48aba438a8dc6de9045d/part2	
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
$__inference_signature_wrapper_632887
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
!__inference__wrapped_model_6324502
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
?
L__inference_classifier_graph_layer_call_and_return_conditional_losses_633021
x?
;sequential_project_matrix_transpose_readvariableop_resource5
1sequential_layer_1_matmul_readvariableop_resource6
2sequential_layer_1_biasadd_readvariableop_resource4
0sequential_output_matmul_readvariableop_resource5
1sequential_output_biasadd_readvariableop_resource
identity??
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
sequential/output/Softmaxw
IdentityIdentity#sequential/output/Softmax:softmax:0*
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
?
?
"__inference__traced_restore_633400
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
?
|
'__inference_output_layer_call_fn_633337

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
B__inference_output_layer_call_and_return_conditional_losses_6325162
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
+__inference_sequential_layer_call_fn_633215

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
F__inference_sequential_layer_call_and_return_conditional_losses_6326022
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
?
?
B__inference_output_layer_call_and_return_conditional_losses_633328

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
?
?
C__inference_layer-1_layer_call_and_return_conditional_losses_632489

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
?
i
(__inference_project_layer_call_fn_632471
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
C__inference_project_layer_call_and_return_conditional_losses_6324632
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
?
?
C__inference_layer-1_layer_call_and_return_conditional_losses_633308

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
?
?
-__inference_functional_1_layer_call_fn_632969

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
GPU 2J 8? *Q
fLRJ
H__inference_functional_1_layer_call_and_return_conditional_losses_6328572
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
H__inference_functional_1_layer_call_and_return_conditional_losses_632809
input_1
classifier_graph_632797
classifier_graph_632799
classifier_graph_632801
classifier_graph_632803
classifier_graph_632805
identity??(classifier_graph/StatefulPartitionedCall?
(classifier_graph/StatefulPartitionedCallStatefulPartitionedCallinput_1classifier_graph_632797classifier_graph_632799classifier_graph_632801classifier_graph_632803classifier_graph_632805*
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
L__inference_classifier_graph_layer_call_and_return_conditional_losses_6326942*
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
?
?
B__inference_output_layer_call_and_return_conditional_losses_632516

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
?
?
F__inference_sequential_layer_call_and_return_conditional_losses_633267
project_input4
0project_matrix_transpose_readvariableop_resource*
&layer_1_matmul_readvariableop_resource+
'layer_1_biasadd_readvariableop_resource)
%output_matmul_readvariableop_resource*
&output_biasadd_readvariableop_resource
identity??
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
output/Softmaxl
IdentityIdentityoutput/Softmax:softmax:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????	::::::V R
'
_output_shapes
:?????????	
'
_user_specified_nameproject_input
?'
?
H__inference_functional_1_layer_call_and_return_conditional_losses_632939

inputsP
Lclassifier_graph_sequential_project_matrix_transpose_readvariableop_resourceF
Bclassifier_graph_sequential_layer_1_matmul_readvariableop_resourceG
Cclassifier_graph_sequential_layer_1_biasadd_readvariableop_resourceE
Aclassifier_graph_sequential_output_matmul_readvariableop_resourceF
Bclassifier_graph_sequential_output_biasadd_readvariableop_resource
identity??
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
*classifier_graph/sequential/output/Softmax?
IdentityIdentity4classifier_graph/sequential/output/Softmax:softmax:0*
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
F__inference_sequential_layer_call_and_return_conditional_losses_633241
project_input4
0project_matrix_transpose_readvariableop_resource*
&layer_1_matmul_readvariableop_resource+
'layer_1_biasadd_readvariableop_resource)
%output_matmul_readvariableop_resource*
&output_biasadd_readvariableop_resource
identity??
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
output/Softmaxl
IdentityIdentityoutput/Softmax:softmax:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????	::::::V R
'
_output_shapes
:?????????	
'
_user_specified_nameproject_input
?-
?
!__inference__wrapped_model_632450
input_1]
Yfunctional_1_classifier_graph_sequential_project_matrix_transpose_readvariableop_resourceS
Ofunctional_1_classifier_graph_sequential_layer_1_matmul_readvariableop_resourceT
Pfunctional_1_classifier_graph_sequential_layer_1_biasadd_readvariableop_resourceR
Nfunctional_1_classifier_graph_sequential_output_matmul_readvariableop_resourceS
Ofunctional_1_classifier_graph_sequential_output_biasadd_readvariableop_resource
identity??
Pfunctional_1/classifier_graph/sequential/project/matrix_transpose/ReadVariableOpReadVariableOpYfunctional_1_classifier_graph_sequential_project_matrix_transpose_readvariableop_resource*
_output_shapes

:	*
dtype02R
Pfunctional_1/classifier_graph/sequential/project/matrix_transpose/ReadVariableOp?
Pfunctional_1/classifier_graph/sequential/project/matrix_transpose/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2R
Pfunctional_1/classifier_graph/sequential/project/matrix_transpose/transpose/perm?
Kfunctional_1/classifier_graph/sequential/project/matrix_transpose/transpose	TransposeXfunctional_1/classifier_graph/sequential/project/matrix_transpose/ReadVariableOp:value:0Yfunctional_1/classifier_graph/sequential/project/matrix_transpose/transpose/perm:output:0*
T0*
_output_shapes

:	2M
Kfunctional_1/classifier_graph/sequential/project/matrix_transpose/transpose?
7functional_1/classifier_graph/sequential/project/matmulMatMulinput_1Ofunctional_1/classifier_graph/sequential/project/matrix_transpose/transpose:y:0*
T0*'
_output_shapes
:?????????29
7functional_1/classifier_graph/sequential/project/matmul?
Hfunctional_1/classifier_graph/sequential/project/matmul_1/ReadVariableOpReadVariableOpYfunctional_1_classifier_graph_sequential_project_matrix_transpose_readvariableop_resource*
_output_shapes

:	*
dtype02J
Hfunctional_1/classifier_graph/sequential/project/matmul_1/ReadVariableOp?
9functional_1/classifier_graph/sequential/project/matmul_1MatMulAfunctional_1/classifier_graph/sequential/project/matmul:product:0Pfunctional_1/classifier_graph/sequential/project/matmul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	2;
9functional_1/classifier_graph/sequential/project/matmul_1?
4functional_1/classifier_graph/sequential/project/subSubinput_1Cfunctional_1/classifier_graph/sequential/project/matmul_1:product:0*
T0*'
_output_shapes
:?????????	26
4functional_1/classifier_graph/sequential/project/sub?
Ffunctional_1/classifier_graph/sequential/layer-1/MatMul/ReadVariableOpReadVariableOpOfunctional_1_classifier_graph_sequential_layer_1_matmul_readvariableop_resource*
_output_shapes

:	2*
dtype02H
Ffunctional_1/classifier_graph/sequential/layer-1/MatMul/ReadVariableOp?
7functional_1/classifier_graph/sequential/layer-1/MatMulMatMul8functional_1/classifier_graph/sequential/project/sub:z:0Nfunctional_1/classifier_graph/sequential/layer-1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????229
7functional_1/classifier_graph/sequential/layer-1/MatMul?
Gfunctional_1/classifier_graph/sequential/layer-1/BiasAdd/ReadVariableOpReadVariableOpPfunctional_1_classifier_graph_sequential_layer_1_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02I
Gfunctional_1/classifier_graph/sequential/layer-1/BiasAdd/ReadVariableOp?
8functional_1/classifier_graph/sequential/layer-1/BiasAddBiasAddAfunctional_1/classifier_graph/sequential/layer-1/MatMul:product:0Ofunctional_1/classifier_graph/sequential/layer-1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22:
8functional_1/classifier_graph/sequential/layer-1/BiasAdd?
5functional_1/classifier_graph/sequential/layer-1/ReluReluAfunctional_1/classifier_graph/sequential/layer-1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????227
5functional_1/classifier_graph/sequential/layer-1/Relu?
Efunctional_1/classifier_graph/sequential/output/MatMul/ReadVariableOpReadVariableOpNfunctional_1_classifier_graph_sequential_output_matmul_readvariableop_resource*
_output_shapes

:2*
dtype02G
Efunctional_1/classifier_graph/sequential/output/MatMul/ReadVariableOp?
6functional_1/classifier_graph/sequential/output/MatMulMatMulCfunctional_1/classifier_graph/sequential/layer-1/Relu:activations:0Mfunctional_1/classifier_graph/sequential/output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????28
6functional_1/classifier_graph/sequential/output/MatMul?
Ffunctional_1/classifier_graph/sequential/output/BiasAdd/ReadVariableOpReadVariableOpOfunctional_1_classifier_graph_sequential_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02H
Ffunctional_1/classifier_graph/sequential/output/BiasAdd/ReadVariableOp?
7functional_1/classifier_graph/sequential/output/BiasAddBiasAdd@functional_1/classifier_graph/sequential/output/MatMul:product:0Nfunctional_1/classifier_graph/sequential/output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????29
7functional_1/classifier_graph/sequential/output/BiasAdd?
7functional_1/classifier_graph/sequential/output/SoftmaxSoftmax@functional_1/classifier_graph/sequential/output/BiasAdd:output:0*
T0*'
_output_shapes
:?????????29
7functional_1/classifier_graph/sequential/output/Softmax?
IdentityIdentityAfunctional_1/classifier_graph/sequential/output/Softmax:softmax:0*
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
?'
?
H__inference_functional_1_layer_call_and_return_conditional_losses_632913

inputsP
Lclassifier_graph_sequential_project_matrix_transpose_readvariableop_resourceF
Bclassifier_graph_sequential_layer_1_matmul_readvariableop_resourceG
Cclassifier_graph_sequential_layer_1_biasadd_readvariableop_resourceE
Aclassifier_graph_sequential_output_matmul_readvariableop_resourceF
Bclassifier_graph_sequential_output_biasadd_readvariableop_resource
identity??
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
*classifier_graph/sequential/output/Softmax?
IdentityIdentity4classifier_graph/sequential/output/Softmax:softmax:0*
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
F__inference_sequential_layer_call_and_return_conditional_losses_633185

inputs4
0project_matrix_transpose_readvariableop_resource*
&layer_1_matmul_readvariableop_resource+
'layer_1_biasadd_readvariableop_resource)
%output_matmul_readvariableop_resource*
&output_biasadd_readvariableop_resource
identity??
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
?

?
C__inference_project_layer_call_and_return_conditional_losses_632463
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
H__inference_functional_1_layer_call_and_return_conditional_losses_632857

inputs
classifier_graph_632845
classifier_graph_632847
classifier_graph_632849
classifier_graph_632851
classifier_graph_632853
identity??(classifier_graph/StatefulPartitionedCall?
(classifier_graph/StatefulPartitionedCallStatefulPartitionedCallinputsclassifier_graph_632845classifier_graph_632847classifier_graph_632849classifier_graph_632851classifier_graph_632853*
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
L__inference_classifier_graph_layer_call_and_return_conditional_losses_6326942*
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
+__inference_sequential_layer_call_fn_633200

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
F__inference_sequential_layer_call_and_return_conditional_losses_6325702
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
+__inference_sequential_layer_call_fn_633297
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
F__inference_sequential_layer_call_and_return_conditional_losses_6326022
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
?
}
(__inference_layer-1_layer_call_fn_633317

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
C__inference_layer-1_layer_call_and_return_conditional_losses_6324892
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
?	
?
H__inference_functional_1_layer_call_and_return_conditional_losses_632794
input_1
classifier_graph_632782
classifier_graph_632784
classifier_graph_632786
classifier_graph_632788
classifier_graph_632790
identity??(classifier_graph/StatefulPartitionedCall?
(classifier_graph/StatefulPartitionedCallStatefulPartitionedCallinput_1classifier_graph_632782classifier_graph_632784classifier_graph_632786classifier_graph_632788classifier_graph_632790*
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
L__inference_classifier_graph_layer_call_and_return_conditional_losses_6327512*
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
?
?
-__inference_functional_1_layer_call_fn_632870
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
GPU 2J 8? *Q
fLRJ
H__inference_functional_1_layer_call_and_return_conditional_losses_6328572
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
-__inference_functional_1_layer_call_fn_632840
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
GPU 2J 8? *Q
fLRJ
H__inference_functional_1_layer_call_and_return_conditional_losses_6328272
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
?
L__inference_classifier_graph_layer_call_and_return_conditional_losses_633077
input_1?
;sequential_project_matrix_transpose_readvariableop_resource5
1sequential_layer_1_matmul_readvariableop_resource6
2sequential_layer_1_biasadd_readvariableop_resource4
0sequential_output_matmul_readvariableop_resource5
1sequential_output_biasadd_readvariableop_resource
identity??
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
sequential/output/Softmaxw
IdentityIdentity#sequential/output/Softmax:softmax:0*
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
?
?
-__inference_functional_1_layer_call_fn_632954

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
GPU 2J 8? *Q
fLRJ
H__inference_functional_1_layer_call_and_return_conditional_losses_6328272
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
?
?
F__inference_sequential_layer_call_and_return_conditional_losses_632570

inputs
project_632556
layer_1_632559
layer_1_632561
output_632564
output_632566
identity??layer-1/StatefulPartitionedCall?output/StatefulPartitionedCall?project/StatefulPartitionedCall?
project/StatefulPartitionedCallStatefulPartitionedCallinputsproject_632556*
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
C__inference_project_layer_call_and_return_conditional_losses_6324632!
project/StatefulPartitionedCall?
layer-1/StatefulPartitionedCallStatefulPartitionedCall(project/StatefulPartitionedCall:output:0layer_1_632559layer_1_632561*
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
C__inference_layer-1_layer_call_and_return_conditional_losses_6324892!
layer-1/StatefulPartitionedCall?
output/StatefulPartitionedCallStatefulPartitionedCall(layer-1/StatefulPartitionedCall:output:0output_632564output_632566*
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
B__inference_output_layer_call_and_return_conditional_losses_6325162 
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
 
_user_specified_nameinputs"?L
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
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
?	
layer-0
layer_with_weights-0
layer-1
trainable_variables
	variables
regularization_losses
	keras_api

signatures
I__call__
*J&call_and_return_all_conditional_losses
K_default_save_signature"?
_tf_keras_network?{"class_name": "Functional", "name": "functional_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "functional_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 9]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "ClassifierGraph", "config": {"layer was saved without config": true}, "name": "classifier_graph", "inbound_nodes": [[["input_1", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["classifier_graph", 0, 0]]}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 9]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional"}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input_1", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 9]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 9]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}}
?

Layers
		model

trainable_variables
	variables
regularization_losses
	keras_api
L__call__
*M&call_and_return_all_conditional_losses"?
_tf_keras_model?{"class_name": "ClassifierGraph", "name": "classifier_graph", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "is_graph_network": false, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "ClassifierGraph"}}
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
 "
trackable_list_wrapper
?
layer_regularization_losses

layers
trainable_variables
	variables
non_trainable_variables
regularization_losses
metrics
layer_metrics
I__call__
K_default_save_signature
*J&call_and_return_all_conditional_losses
&J"call_and_return_conditional_losses"
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
trainable_variables
	variables
regularization_losses
	keras_api
O__call__
*P&call_and_return_all_conditional_losses"?
_tf_keras_sequential?{"class_name": "Sequential", "name": "sequential", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 9]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "project_input"}}, {"class_name": "Project", "config": {"layer was saved without config": true}}, {"class_name": "Dense", "config": {"name": "layer-1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 9]}, "dtype": "float32", "units": 50, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "output", "trainable": true, "dtype": "float32", "units": 2, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 9]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 9]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential"}}
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
 "
trackable_list_wrapper
?
layer_regularization_losses

 layers

trainable_variables
	variables
!non_trainable_variables
regularization_losses
"metrics
#layer_metrics
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
.
0
1"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?
w
$_inbound_nodes
%_outbound_nodes
&trainable_variables
'	variables
(regularization_losses
)	keras_api
Q__call__
*R&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Project", "name": "project", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 9]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?	
*_inbound_nodes

kernel
bias
+_outbound_nodes
,trainable_variables
-	variables
.regularization_losses
/	keras_api
S__call__
*T&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "layer-1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 9]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "layer-1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 9]}, "dtype": "float32", "units": 50, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 9}}}, "build_input_shape": {"class_name": "TensorShape", "items": [250, 9]}}
?
0_inbound_nodes

kernel
bias
1trainable_variables
2	variables
3regularization_losses
4	keras_api
U__call__
*V&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "output", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "output", "trainable": true, "dtype": "float32", "units": 2, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 50}}}, "build_input_shape": {"class_name": "TensorShape", "items": [250, 50]}}
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
 "
trackable_list_wrapper
?
5layer_regularization_losses

6layers
trainable_variables
	variables
7non_trainable_variables
regularization_losses
8metrics
9layer_metrics
O__call__
*P&call_and_return_all_conditional_losses
&P"call_and_return_conditional_losses"
_generic_user_object
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
 "
trackable_dict_wrapper
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
?
:layer_regularization_losses

;layers
&trainable_variables
'	variables
<non_trainable_variables
(regularization_losses
=metrics
>layer_metrics
Q__call__
*R&call_and_return_all_conditional_losses
&R"call_and_return_conditional_losses"
_generic_user_object
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
 "
trackable_list_wrapper
?
?layer_regularization_losses

@layers
,trainable_variables
-	variables
Anon_trainable_variables
.regularization_losses
Bmetrics
Clayer_metrics
S__call__
*T&call_and_return_all_conditional_losses
&T"call_and_return_conditional_losses"
_generic_user_object
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
 "
trackable_list_wrapper
?
Dlayer_regularization_losses

Elayers
1trainable_variables
2	variables
Fnon_trainable_variables
3regularization_losses
Gmetrics
Hlayer_metrics
U__call__
*V&call_and_return_all_conditional_losses
&V"call_and_return_conditional_losses"
_generic_user_object
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
 "
trackable_dict_wrapper
?2?
-__inference_functional_1_layer_call_fn_632870
-__inference_functional_1_layer_call_fn_632954
-__inference_functional_1_layer_call_fn_632969
-__inference_functional_1_layer_call_fn_632840?
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
H__inference_functional_1_layer_call_and_return_conditional_losses_632939
H__inference_functional_1_layer_call_and_return_conditional_losses_632913
H__inference_functional_1_layer_call_and_return_conditional_losses_632809
H__inference_functional_1_layer_call_and_return_conditional_losses_632794?
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
!__inference__wrapped_model_632450?
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
1__inference_classifier_graph_layer_call_fn_633036
1__inference_classifier_graph_layer_call_fn_633133
1__inference_classifier_graph_layer_call_fn_633051
1__inference_classifier_graph_layer_call_fn_633118?
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
L__inference_classifier_graph_layer_call_and_return_conditional_losses_633103
L__inference_classifier_graph_layer_call_and_return_conditional_losses_632995
L__inference_classifier_graph_layer_call_and_return_conditional_losses_633077
L__inference_classifier_graph_layer_call_and_return_conditional_losses_633021?
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
3B1
$__inference_signature_wrapper_632887input_1
?2?
+__inference_sequential_layer_call_fn_633200
+__inference_sequential_layer_call_fn_633297
+__inference_sequential_layer_call_fn_633215
+__inference_sequential_layer_call_fn_633282?
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
F__inference_sequential_layer_call_and_return_conditional_losses_633241
F__inference_sequential_layer_call_and_return_conditional_losses_633159
F__inference_sequential_layer_call_and_return_conditional_losses_633185
F__inference_sequential_layer_call_and_return_conditional_losses_633267?
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
(__inference_project_layer_call_fn_632471?
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
C__inference_project_layer_call_and_return_conditional_losses_632463?
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
(__inference_layer-1_layer_call_fn_633317?
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
C__inference_layer-1_layer_call_and_return_conditional_losses_633308?
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
'__inference_output_layer_call_fn_633337?
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
B__inference_output_layer_call_and_return_conditional_losses_633328?
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
!__inference__wrapped_model_632450~0?-
&?#
!?
input_1?????????	
? "C?@
>
classifier_graph*?'
classifier_graph??????????
L__inference_classifier_graph_layer_call_and_return_conditional_losses_632995b2?/
(?%
?
x?????????	
p 
p
? "%?"
?
0?????????
? ?
L__inference_classifier_graph_layer_call_and_return_conditional_losses_633021b2?/
(?%
?
x?????????	
p 
p 
? "%?"
?
0?????????
? ?
L__inference_classifier_graph_layer_call_and_return_conditional_losses_633077h8?5
.?+
!?
input_1?????????	
p 
p
? "%?"
?
0?????????
? ?
L__inference_classifier_graph_layer_call_and_return_conditional_losses_633103h8?5
.?+
!?
input_1?????????	
p 
p 
? "%?"
?
0?????????
? ?
1__inference_classifier_graph_layer_call_fn_633036U2?/
(?%
?
x?????????	
p 
p
? "???????????
1__inference_classifier_graph_layer_call_fn_633051U2?/
(?%
?
x?????????	
p 
p 
? "???????????
1__inference_classifier_graph_layer_call_fn_633118[8?5
.?+
!?
input_1?????????	
p 
p
? "???????????
1__inference_classifier_graph_layer_call_fn_633133[8?5
.?+
!?
input_1?????????	
p 
p 
? "???????????
H__inference_functional_1_layer_call_and_return_conditional_losses_632794h8?5
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
H__inference_functional_1_layer_call_and_return_conditional_losses_632809h8?5
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
H__inference_functional_1_layer_call_and_return_conditional_losses_632913g7?4
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
H__inference_functional_1_layer_call_and_return_conditional_losses_632939g7?4
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
-__inference_functional_1_layer_call_fn_632840[8?5
.?+
!?
input_1?????????	
p

 
? "???????????
-__inference_functional_1_layer_call_fn_632870[8?5
.?+
!?
input_1?????????	
p 

 
? "???????????
-__inference_functional_1_layer_call_fn_632954Z7?4
-?*
 ?
inputs?????????	
p

 
? "???????????
-__inference_functional_1_layer_call_fn_632969Z7?4
-?*
 ?
inputs?????????	
p 

 
? "???????????
C__inference_layer-1_layer_call_and_return_conditional_losses_633308\/?,
%?"
 ?
inputs?????????	
? "%?"
?
0?????????2
? {
(__inference_layer-1_layer_call_fn_633317O/?,
%?"
 ?
inputs?????????	
? "??????????2?
B__inference_output_layer_call_and_return_conditional_losses_633328\/?,
%?"
 ?
inputs?????????2
? "%?"
?
0?????????
? z
'__inference_output_layer_call_fn_633337O/?,
%?"
 ?
inputs?????????2
? "???????????
C__inference_project_layer_call_and_return_conditional_losses_632463V*?'
 ?
?
x?????????	
? "%?"
?
0?????????	
? u
(__inference_project_layer_call_fn_632471I*?'
 ?
?
x?????????	
? "??????????	?
F__inference_sequential_layer_call_and_return_conditional_losses_633159g7?4
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
F__inference_sequential_layer_call_and_return_conditional_losses_633185g7?4
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
F__inference_sequential_layer_call_and_return_conditional_losses_633241n>?;
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
F__inference_sequential_layer_call_and_return_conditional_losses_633267n>?;
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
+__inference_sequential_layer_call_fn_633200Z7?4
-?*
 ?
inputs?????????	
p

 
? "???????????
+__inference_sequential_layer_call_fn_633215Z7?4
-?*
 ?
inputs?????????	
p 

 
? "???????????
+__inference_sequential_layer_call_fn_633282a>?;
4?1
'?$
project_input?????????	
p

 
? "???????????
+__inference_sequential_layer_call_fn_633297a>?;
4?1
'?$
project_input?????????	
p 

 
? "???????????
$__inference_signature_wrapper_632887?;?8
? 
1?.
,
input_1!?
input_1?????????	"C?@
>
classifier_graph*?'
classifier_graph?????????