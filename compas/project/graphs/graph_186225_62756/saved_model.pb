гђ
ЛБ
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
dtypetypeѕ
Й
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
executor_typestring ѕ
ќ
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ѕ"serve*2.3.02v2.3.0-rc2-23-gb36436b0878оо
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
ф
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*т
value█Bп BЛ
ќ
layer-0
layer_with_weights-0
layer-1
regularization_losses
	variables
trainable_variables
	keras_api

signatures
 
i

Layers
		model

regularization_losses
	variables
trainable_variables
	keras_api
 
#
0
1
2
3
4

0
1
2
3
Г
regularization_losses
layer_regularization_losses
non_trainable_variables
	variables
metrics
layer_metrics
trainable_variables

layers
 

0
1
2
К
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
regularization_losses
	variables
trainable_variables
	keras_api
 
#
0
1
2
3
4

0
1
2
3
Г

regularization_losses
layer_regularization_losses
 non_trainable_variables
	variables
!metrics
"layer_metrics
trainable_variables

#layers
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

0
 
 

0
1
ѓ
w
$_inbound_nodes
%_outbound_nodes
&regularization_losses
'	variables
(trainable_variables
)	keras_api
Љ
*_inbound_nodes

kernel
bias
+_outbound_nodes
,regularization_losses
-	variables
.trainable_variables
/	keras_api
|
0_inbound_nodes

kernel
bias
1regularization_losses
2	variables
3trainable_variables
4	keras_api
 
#
0
1
2
3
4

0
1
2
3
Г
regularization_losses
5layer_regularization_losses
6non_trainable_variables
	variables
7metrics
8layer_metrics
trainable_variables

9layers
 

0
 
 

0
1
2
	3
 
 
 

0
 
Г
&regularization_losses
:layer_regularization_losses
;non_trainable_variables
'	variables
<metrics
=layer_metrics
(trainable_variables

>layers
 
 
 

0
1

0
1
Г
,regularization_losses
?layer_regularization_losses
@non_trainable_variables
-	variables
Ametrics
Blayer_metrics
.trainable_variables

Clayers
 
 

0
1

0
1
Г
1regularization_losses
Dlayer_regularization_losses
Enon_trainable_variables
2	variables
Fmetrics
Glayer_metrics
3trainable_variables

Hlayers
 

0
 
 

0
1
2
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
 
z
serving_default_input_1Placeholder*'
_output_shapes
:         	*
dtype0*
shape:         	
ђ
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1Variablelayer-1/kernellayer-1/biasoutput/kerneloutput/bias*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8ѓ *-
f(R&
$__inference_signature_wrapper_632887
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
╚
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
GPU 2J 8ѓ *(
f#R!
__inference__traced_save_633375
▀
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
GPU 2J 8ѓ *+
f&R$
"__inference__traced_restore_633400▓г
в
г
F__inference_sequential_layer_call_and_return_conditional_losses_632602

inputs
project_632588
layer_1_632591
layer_1_632593
output_632596
output_632598
identityѕбlayer-1/StatefulPartitionedCallбoutput/StatefulPartitionedCallбproject/StatefulPartitionedCall§
project/StatefulPartitionedCallStatefulPartitionedCallinputsproject_632588*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         	*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_project_layer_call_and_return_conditional_losses_6324632!
project/StatefulPartitionedCall▒
layer-1/StatefulPartitionedCallStatefulPartitionedCall(project/StatefulPartitionedCall:output:0layer_1_632591layer_1_632593*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_layer-1_layer_call_and_return_conditional_losses_6324892!
layer-1/StatefulPartitionedCallг
output/StatefulPartitionedCallStatefulPartitionedCall(layer-1/StatefulPartitionedCall:output:0output_632596output_632598*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *K
fFRD
B__inference_output_layer_call_and_return_conditional_losses_6325162 
output/StatefulPartitionedCallЯ
IdentityIdentity'output/StatefulPartitionedCall:output:0 ^layer-1/StatefulPartitionedCall^output/StatefulPartitionedCall ^project/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*:
_input_shapes)
':         	:::::2B
layer-1/StatefulPartitionedCalllayer-1/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall2B
project/StatefulPartitionedCallproject/StatefulPartitionedCall:O K
'
_output_shapes
:         	
 
_user_specified_nameinputs
ў
╦
F__inference_sequential_layer_call_and_return_conditional_losses_633159

inputs4
0project_matrix_transpose_readvariableop_resource*
&layer_1_matmul_readvariableop_resource+
'layer_1_biasadd_readvariableop_resource)
%output_matmul_readvariableop_resource*
&output_biasadd_readvariableop_resource
identityѕ├
'project/matrix_transpose/ReadVariableOpReadVariableOp0project_matrix_transpose_readvariableop_resource*
_output_shapes

:	*
dtype02)
'project/matrix_transpose/ReadVariableOpБ
'project/matrix_transpose/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2)
'project/matrix_transpose/transpose/permр
"project/matrix_transpose/transpose	Transpose/project/matrix_transpose/ReadVariableOp:value:00project/matrix_transpose/transpose/perm:output:0*
T0*
_output_shapes

:	2$
"project/matrix_transpose/transposeї
project/matmulMatMulinputs&project/matrix_transpose/transpose:y:0*
T0*'
_output_shapes
:         2
project/matmul│
project/matmul_1/ReadVariableOpReadVariableOp0project_matrix_transpose_readvariableop_resource*
_output_shapes

:	*
dtype02!
project/matmul_1/ReadVariableOpБ
project/matmul_1MatMulproject/matmul:product:0'project/matmul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         	2
project/matmul_1w
project/subSubinputsproject/matmul_1:product:0*
T0*'
_output_shapes
:         	2
project/subЦ
layer-1/MatMul/ReadVariableOpReadVariableOp&layer_1_matmul_readvariableop_resource*
_output_shapes

:	2*
dtype02
layer-1/MatMul/ReadVariableOpћ
layer-1/MatMulMatMulproject/sub:z:0%layer-1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         22
layer-1/MatMulц
layer-1/BiasAdd/ReadVariableOpReadVariableOp'layer_1_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02 
layer-1/BiasAdd/ReadVariableOpА
layer-1/BiasAddBiasAddlayer-1/MatMul:product:0&layer-1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         22
layer-1/BiasAddp
layer-1/ReluRelulayer-1/BiasAdd:output:0*
T0*'
_output_shapes
:         22
layer-1/Reluб
output/MatMul/ReadVariableOpReadVariableOp%output_matmul_readvariableop_resource*
_output_shapes

:2*
dtype02
output/MatMul/ReadVariableOpю
output/MatMulMatMullayer-1/Relu:activations:0$output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
output/MatMulА
output/BiasAdd/ReadVariableOpReadVariableOp&output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
output/BiasAdd/ReadVariableOpЮ
output/BiasAddBiasAddoutput/MatMul:product:0%output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
output/BiasAddv
output/SoftmaxSoftmaxoutput/BiasAdd:output:0*
T0*'
_output_shapes
:         2
output/Softmaxl
IdentityIdentityoutput/Softmax:softmax:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*:
_input_shapes)
':         	::::::O K
'
_output_shapes
:         	
 
_user_specified_nameinputs
л
┤
1__inference_classifier_graph_layer_call_fn_633036
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identityѕбStatefulPartitionedCallц
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8ѓ *U
fPRN
L__inference_classifier_graph_layer_call_and_return_conditional_losses_6326942
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*:
_input_shapes)
':         	:::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:         	
!
_user_specified_name	input_1
Й
«
1__inference_classifier_graph_layer_call_fn_633118
x
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identityѕбStatefulPartitionedCallъ
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8ѓ *U
fPRN
L__inference_classifier_graph_layer_call_and_return_conditional_losses_6326942
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*:
_input_shapes)
':         	:::::22
StatefulPartitionedCallStatefulPartitionedCall:J F
'
_output_shapes
:         	

_user_specified_namex
о
┤
+__inference_sequential_layer_call_fn_633282
project_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identityѕбStatefulPartitionedCallц
StatefulPartitionedCallStatefulPartitionedCallproject_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_6325702
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*:
_input_shapes)
':         	:::::22
StatefulPartitionedCallStatefulPartitionedCall:V R
'
_output_shapes
:         	
'
_user_specified_nameproject_input
в	
Б
H__inference_functional_1_layer_call_and_return_conditional_losses_632827

inputs
classifier_graph_632815
classifier_graph_632817
classifier_graph_632819
classifier_graph_632821
classifier_graph_632823
identityѕб(classifier_graph/StatefulPartitionedCallЇ
(classifier_graph/StatefulPartitionedCallStatefulPartitionedCallinputsclassifier_graph_632815classifier_graph_632817classifier_graph_632819classifier_graph_632821classifier_graph_632823*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8ѓ *U
fPRN
L__inference_classifier_graph_layer_call_and_return_conditional_losses_6327512*
(classifier_graph/StatefulPartitionedCall░
IdentityIdentity1classifier_graph/StatefulPartitionedCall:output:0)^classifier_graph/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*:
_input_shapes)
':         	:::::2T
(classifier_graph/StatefulPartitionedCall(classifier_graph/StatefulPartitionedCall:O K
'
_output_shapes
:         	
 
_user_specified_nameinputs
¤
Ѕ
L__inference_classifier_graph_layer_call_and_return_conditional_losses_633021
input_1?
;sequential_project_matrix_transpose_readvariableop_resource5
1sequential_layer_1_matmul_readvariableop_resource6
2sequential_layer_1_biasadd_readvariableop_resource4
0sequential_output_matmul_readvariableop_resource5
1sequential_output_biasadd_readvariableop_resource
identityѕС
2sequential/project/matrix_transpose/ReadVariableOpReadVariableOp;sequential_project_matrix_transpose_readvariableop_resource*
_output_shapes

:	*
dtype024
2sequential/project/matrix_transpose/ReadVariableOp╣
2sequential/project/matrix_transpose/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       24
2sequential/project/matrix_transpose/transpose/permЇ
-sequential/project/matrix_transpose/transpose	Transpose:sequential/project/matrix_transpose/ReadVariableOp:value:0;sequential/project/matrix_transpose/transpose/perm:output:0*
T0*
_output_shapes

:	2/
-sequential/project/matrix_transpose/transpose«
sequential/project/matmulMatMulinput_11sequential/project/matrix_transpose/transpose:y:0*
T0*'
_output_shapes
:         2
sequential/project/matmulн
*sequential/project/matmul_1/ReadVariableOpReadVariableOp;sequential_project_matrix_transpose_readvariableop_resource*
_output_shapes

:	*
dtype02,
*sequential/project/matmul_1/ReadVariableOp¤
sequential/project/matmul_1MatMul#sequential/project/matmul:product:02sequential/project/matmul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         	2
sequential/project/matmul_1Ў
sequential/project/subSubinput_1%sequential/project/matmul_1:product:0*
T0*'
_output_shapes
:         	2
sequential/project/subк
(sequential/layer-1/MatMul/ReadVariableOpReadVariableOp1sequential_layer_1_matmul_readvariableop_resource*
_output_shapes

:	2*
dtype02*
(sequential/layer-1/MatMul/ReadVariableOp└
sequential/layer-1/MatMulMatMulsequential/project/sub:z:00sequential/layer-1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         22
sequential/layer-1/MatMul┼
)sequential/layer-1/BiasAdd/ReadVariableOpReadVariableOp2sequential_layer_1_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02+
)sequential/layer-1/BiasAdd/ReadVariableOp═
sequential/layer-1/BiasAddBiasAdd#sequential/layer-1/MatMul:product:01sequential/layer-1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         22
sequential/layer-1/BiasAddЉ
sequential/layer-1/ReluRelu#sequential/layer-1/BiasAdd:output:0*
T0*'
_output_shapes
:         22
sequential/layer-1/Relu├
'sequential/output/MatMul/ReadVariableOpReadVariableOp0sequential_output_matmul_readvariableop_resource*
_output_shapes

:2*
dtype02)
'sequential/output/MatMul/ReadVariableOp╚
sequential/output/MatMulMatMul%sequential/layer-1/Relu:activations:0/sequential/output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
sequential/output/MatMul┬
(sequential/output/BiasAdd/ReadVariableOpReadVariableOp1sequential_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(sequential/output/BiasAdd/ReadVariableOp╔
sequential/output/BiasAddBiasAdd"sequential/output/MatMul:product:00sequential/output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
sequential/output/BiasAddЌ
sequential/output/SoftmaxSoftmax"sequential/output/BiasAdd:output:0*
T0*'
_output_shapes
:         2
sequential/output/Softmaxw
IdentityIdentity#sequential/output/Softmax:softmax:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*:
_input_shapes)
':         	::::::P L
'
_output_shapes
:         	
!
_user_specified_name	input_1
З
■
L__inference_classifier_graph_layer_call_and_return_conditional_losses_632694
x
sequential_632682
sequential_632684
sequential_632686
sequential_632688
sequential_632690
identityѕб"sequential/StatefulPartitionedCallп
"sequential/StatefulPartitionedCallStatefulPartitionedCallxsequential_632682sequential_632684sequential_632686sequential_632688sequential_632690*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_6326022$
"sequential/StatefulPartitionedCallц
IdentityIdentity+sequential/StatefulPartitionedCall:output:0#^sequential/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*:
_input_shapes)
':         	:::::2H
"sequential/StatefulPartitionedCall"sequential/StatefulPartitionedCall:J F
'
_output_shapes
:         	

_user_specified_namex
и
Ѓ
L__inference_classifier_graph_layer_call_and_return_conditional_losses_632751
x?
;sequential_project_matrix_transpose_readvariableop_resource5
1sequential_layer_1_matmul_readvariableop_resource6
2sequential_layer_1_biasadd_readvariableop_resource4
0sequential_output_matmul_readvariableop_resource5
1sequential_output_biasadd_readvariableop_resource
identityѕС
2sequential/project/matrix_transpose/ReadVariableOpReadVariableOp;sequential_project_matrix_transpose_readvariableop_resource*
_output_shapes

:	*
dtype024
2sequential/project/matrix_transpose/ReadVariableOp╣
2sequential/project/matrix_transpose/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       24
2sequential/project/matrix_transpose/transpose/permЇ
-sequential/project/matrix_transpose/transpose	Transpose:sequential/project/matrix_transpose/ReadVariableOp:value:0;sequential/project/matrix_transpose/transpose/perm:output:0*
T0*
_output_shapes

:	2/
-sequential/project/matrix_transpose/transposeе
sequential/project/matmulMatMulx1sequential/project/matrix_transpose/transpose:y:0*
T0*'
_output_shapes
:         2
sequential/project/matmulн
*sequential/project/matmul_1/ReadVariableOpReadVariableOp;sequential_project_matrix_transpose_readvariableop_resource*
_output_shapes

:	*
dtype02,
*sequential/project/matmul_1/ReadVariableOp¤
sequential/project/matmul_1MatMul#sequential/project/matmul:product:02sequential/project/matmul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         	2
sequential/project/matmul_1Њ
sequential/project/subSubx%sequential/project/matmul_1:product:0*
T0*'
_output_shapes
:         	2
sequential/project/subк
(sequential/layer-1/MatMul/ReadVariableOpReadVariableOp1sequential_layer_1_matmul_readvariableop_resource*
_output_shapes

:	2*
dtype02*
(sequential/layer-1/MatMul/ReadVariableOp└
sequential/layer-1/MatMulMatMulsequential/project/sub:z:00sequential/layer-1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         22
sequential/layer-1/MatMul┼
)sequential/layer-1/BiasAdd/ReadVariableOpReadVariableOp2sequential_layer_1_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02+
)sequential/layer-1/BiasAdd/ReadVariableOp═
sequential/layer-1/BiasAddBiasAdd#sequential/layer-1/MatMul:product:01sequential/layer-1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         22
sequential/layer-1/BiasAddЉ
sequential/layer-1/ReluRelu#sequential/layer-1/BiasAdd:output:0*
T0*'
_output_shapes
:         22
sequential/layer-1/Relu├
'sequential/output/MatMul/ReadVariableOpReadVariableOp0sequential_output_matmul_readvariableop_resource*
_output_shapes

:2*
dtype02)
'sequential/output/MatMul/ReadVariableOp╚
sequential/output/MatMulMatMul%sequential/layer-1/Relu:activations:0/sequential/output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
sequential/output/MatMul┬
(sequential/output/BiasAdd/ReadVariableOpReadVariableOp1sequential_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(sequential/output/BiasAdd/ReadVariableOp╔
sequential/output/BiasAddBiasAdd"sequential/output/MatMul:product:00sequential/output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
sequential/output/BiasAddЌ
sequential/output/SoftmaxSoftmax"sequential/output/BiasAdd:output:0*
T0*'
_output_shapes
:         2
sequential/output/Softmaxw
IdentityIdentity#sequential/output/Softmax:softmax:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*:
_input_shapes)
':         	::::::J F
'
_output_shapes
:         	

_user_specified_namex
¤
Ѕ
L__inference_classifier_graph_layer_call_and_return_conditional_losses_632995
input_1?
;sequential_project_matrix_transpose_readvariableop_resource5
1sequential_layer_1_matmul_readvariableop_resource6
2sequential_layer_1_biasadd_readvariableop_resource4
0sequential_output_matmul_readvariableop_resource5
1sequential_output_biasadd_readvariableop_resource
identityѕС
2sequential/project/matrix_transpose/ReadVariableOpReadVariableOp;sequential_project_matrix_transpose_readvariableop_resource*
_output_shapes

:	*
dtype024
2sequential/project/matrix_transpose/ReadVariableOp╣
2sequential/project/matrix_transpose/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       24
2sequential/project/matrix_transpose/transpose/permЇ
-sequential/project/matrix_transpose/transpose	Transpose:sequential/project/matrix_transpose/ReadVariableOp:value:0;sequential/project/matrix_transpose/transpose/perm:output:0*
T0*
_output_shapes

:	2/
-sequential/project/matrix_transpose/transpose«
sequential/project/matmulMatMulinput_11sequential/project/matrix_transpose/transpose:y:0*
T0*'
_output_shapes
:         2
sequential/project/matmulн
*sequential/project/matmul_1/ReadVariableOpReadVariableOp;sequential_project_matrix_transpose_readvariableop_resource*
_output_shapes

:	*
dtype02,
*sequential/project/matmul_1/ReadVariableOp¤
sequential/project/matmul_1MatMul#sequential/project/matmul:product:02sequential/project/matmul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         	2
sequential/project/matmul_1Ў
sequential/project/subSubinput_1%sequential/project/matmul_1:product:0*
T0*'
_output_shapes
:         	2
sequential/project/subк
(sequential/layer-1/MatMul/ReadVariableOpReadVariableOp1sequential_layer_1_matmul_readvariableop_resource*
_output_shapes

:	2*
dtype02*
(sequential/layer-1/MatMul/ReadVariableOp└
sequential/layer-1/MatMulMatMulsequential/project/sub:z:00sequential/layer-1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         22
sequential/layer-1/MatMul┼
)sequential/layer-1/BiasAdd/ReadVariableOpReadVariableOp2sequential_layer_1_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02+
)sequential/layer-1/BiasAdd/ReadVariableOp═
sequential/layer-1/BiasAddBiasAdd#sequential/layer-1/MatMul:product:01sequential/layer-1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         22
sequential/layer-1/BiasAddЉ
sequential/layer-1/ReluRelu#sequential/layer-1/BiasAdd:output:0*
T0*'
_output_shapes
:         22
sequential/layer-1/Relu├
'sequential/output/MatMul/ReadVariableOpReadVariableOp0sequential_output_matmul_readvariableop_resource*
_output_shapes

:2*
dtype02)
'sequential/output/MatMul/ReadVariableOp╚
sequential/output/MatMulMatMul%sequential/layer-1/Relu:activations:0/sequential/output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
sequential/output/MatMul┬
(sequential/output/BiasAdd/ReadVariableOpReadVariableOp1sequential_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(sequential/output/BiasAdd/ReadVariableOp╔
sequential/output/BiasAddBiasAdd"sequential/output/MatMul:product:00sequential/output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
sequential/output/BiasAddЌ
sequential/output/SoftmaxSoftmax"sequential/output/BiasAdd:output:0*
T0*'
_output_shapes
:         2
sequential/output/Softmaxw
IdentityIdentity#sequential/output/Softmax:softmax:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*:
_input_shapes)
':         	::::::P L
'
_output_shapes
:         	
!
_user_specified_name	input_1
Й
«
1__inference_classifier_graph_layer_call_fn_633133
x
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identityѕбStatefulPartitionedCallъ
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8ѓ *U
fPRN
L__inference_classifier_graph_layer_call_and_return_conditional_losses_6326942
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*:
_input_shapes)
':         	:::::22
StatefulPartitionedCallStatefulPartitionedCall:J F
'
_output_shapes
:         	

_user_specified_namex
и
Ѓ
L__inference_classifier_graph_layer_call_and_return_conditional_losses_633103
x?
;sequential_project_matrix_transpose_readvariableop_resource5
1sequential_layer_1_matmul_readvariableop_resource6
2sequential_layer_1_biasadd_readvariableop_resource4
0sequential_output_matmul_readvariableop_resource5
1sequential_output_biasadd_readvariableop_resource
identityѕС
2sequential/project/matrix_transpose/ReadVariableOpReadVariableOp;sequential_project_matrix_transpose_readvariableop_resource*
_output_shapes

:	*
dtype024
2sequential/project/matrix_transpose/ReadVariableOp╣
2sequential/project/matrix_transpose/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       24
2sequential/project/matrix_transpose/transpose/permЇ
-sequential/project/matrix_transpose/transpose	Transpose:sequential/project/matrix_transpose/ReadVariableOp:value:0;sequential/project/matrix_transpose/transpose/perm:output:0*
T0*
_output_shapes

:	2/
-sequential/project/matrix_transpose/transposeе
sequential/project/matmulMatMulx1sequential/project/matrix_transpose/transpose:y:0*
T0*'
_output_shapes
:         2
sequential/project/matmulн
*sequential/project/matmul_1/ReadVariableOpReadVariableOp;sequential_project_matrix_transpose_readvariableop_resource*
_output_shapes

:	*
dtype02,
*sequential/project/matmul_1/ReadVariableOp¤
sequential/project/matmul_1MatMul#sequential/project/matmul:product:02sequential/project/matmul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         	2
sequential/project/matmul_1Њ
sequential/project/subSubx%sequential/project/matmul_1:product:0*
T0*'
_output_shapes
:         	2
sequential/project/subк
(sequential/layer-1/MatMul/ReadVariableOpReadVariableOp1sequential_layer_1_matmul_readvariableop_resource*
_output_shapes

:	2*
dtype02*
(sequential/layer-1/MatMul/ReadVariableOp└
sequential/layer-1/MatMulMatMulsequential/project/sub:z:00sequential/layer-1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         22
sequential/layer-1/MatMul┼
)sequential/layer-1/BiasAdd/ReadVariableOpReadVariableOp2sequential_layer_1_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02+
)sequential/layer-1/BiasAdd/ReadVariableOp═
sequential/layer-1/BiasAddBiasAdd#sequential/layer-1/MatMul:product:01sequential/layer-1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         22
sequential/layer-1/BiasAddЉ
sequential/layer-1/ReluRelu#sequential/layer-1/BiasAdd:output:0*
T0*'
_output_shapes
:         22
sequential/layer-1/Relu├
'sequential/output/MatMul/ReadVariableOpReadVariableOp0sequential_output_matmul_readvariableop_resource*
_output_shapes

:2*
dtype02)
'sequential/output/MatMul/ReadVariableOp╚
sequential/output/MatMulMatMul%sequential/layer-1/Relu:activations:0/sequential/output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
sequential/output/MatMul┬
(sequential/output/BiasAdd/ReadVariableOpReadVariableOp1sequential_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(sequential/output/BiasAdd/ReadVariableOp╔
sequential/output/BiasAddBiasAdd"sequential/output/MatMul:product:00sequential/output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
sequential/output/BiasAddЌ
sequential/output/SoftmaxSoftmax"sequential/output/BiasAdd:output:0*
T0*'
_output_shapes
:         2
sequential/output/Softmaxw
IdentityIdentity#sequential/output/Softmax:softmax:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*:
_input_shapes)
':         	::::::J F
'
_output_shapes
:         	

_user_specified_namex
л
┤
1__inference_classifier_graph_layer_call_fn_633051
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identityѕбStatefulPartitionedCallц
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8ѓ *U
fPRN
L__inference_classifier_graph_layer_call_and_return_conditional_losses_6326942
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*:
_input_shapes)
':         	:::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:         	
!
_user_specified_name	input_1
┌
}
(__inference_layer-1_layer_call_fn_633317

inputs
unknown
	unknown_0
identityѕбStatefulPartitionedCallз
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_layer-1_layer_call_and_return_conditional_losses_6324892
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         22

Identity"
identityIdentity:output:0*.
_input_shapes
:         	::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         	
 
_user_specified_nameinputs
ў
Д
$__inference_signature_wrapper_632887
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identityѕбStatefulPartitionedCallщ
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8ѓ **
f%R#
!__inference__wrapped_model_6324502
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*:
_input_shapes)
':         	:::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:         	
!
_user_specified_name	input_1
ў
╦
F__inference_sequential_layer_call_and_return_conditional_losses_633185

inputs4
0project_matrix_transpose_readvariableop_resource*
&layer_1_matmul_readvariableop_resource+
'layer_1_biasadd_readvariableop_resource)
%output_matmul_readvariableop_resource*
&output_biasadd_readvariableop_resource
identityѕ├
'project/matrix_transpose/ReadVariableOpReadVariableOp0project_matrix_transpose_readvariableop_resource*
_output_shapes

:	*
dtype02)
'project/matrix_transpose/ReadVariableOpБ
'project/matrix_transpose/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2)
'project/matrix_transpose/transpose/permр
"project/matrix_transpose/transpose	Transpose/project/matrix_transpose/ReadVariableOp:value:00project/matrix_transpose/transpose/perm:output:0*
T0*
_output_shapes

:	2$
"project/matrix_transpose/transposeї
project/matmulMatMulinputs&project/matrix_transpose/transpose:y:0*
T0*'
_output_shapes
:         2
project/matmul│
project/matmul_1/ReadVariableOpReadVariableOp0project_matrix_transpose_readvariableop_resource*
_output_shapes

:	*
dtype02!
project/matmul_1/ReadVariableOpБ
project/matmul_1MatMulproject/matmul:product:0'project/matmul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         	2
project/matmul_1w
project/subSubinputsproject/matmul_1:product:0*
T0*'
_output_shapes
:         	2
project/subЦ
layer-1/MatMul/ReadVariableOpReadVariableOp&layer_1_matmul_readvariableop_resource*
_output_shapes

:	2*
dtype02
layer-1/MatMul/ReadVariableOpћ
layer-1/MatMulMatMulproject/sub:z:0%layer-1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         22
layer-1/MatMulц
layer-1/BiasAdd/ReadVariableOpReadVariableOp'layer_1_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02 
layer-1/BiasAdd/ReadVariableOpА
layer-1/BiasAddBiasAddlayer-1/MatMul:product:0&layer-1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         22
layer-1/BiasAddp
layer-1/ReluRelulayer-1/BiasAdd:output:0*
T0*'
_output_shapes
:         22
layer-1/Reluб
output/MatMul/ReadVariableOpReadVariableOp%output_matmul_readvariableop_resource*
_output_shapes

:2*
dtype02
output/MatMul/ReadVariableOpю
output/MatMulMatMullayer-1/Relu:activations:0$output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
output/MatMulА
output/BiasAdd/ReadVariableOpReadVariableOp&output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
output/BiasAdd/ReadVariableOpЮ
output/BiasAddBiasAddoutput/MatMul:product:0%output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
output/BiasAddv
output/SoftmaxSoftmaxoutput/BiasAdd:output:0*
T0*'
_output_shapes
:         2
output/Softmaxl
IdentityIdentityoutput/Softmax:softmax:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*:
_input_shapes)
':         	::::::O K
'
_output_shapes
:         	
 
_user_specified_nameinputs
▄
╦
__inference__traced_save_633375
file_prefix-
)savev2_layer_1_kernel_read_readvariableop+
'savev2_layer_1_bias_read_readvariableop,
(savev2_output_kernel_read_readvariableop*
&savev2_output_bias_read_readvariableop'
#savev2_variable_read_readvariableop
savev2_const

identity_1ѕбMergeV2CheckpointsЈ
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
ConstЇ
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_e61ac41907a94ba98b03296b5ea03b02/part2	
Const_1І
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
ShardedFilename/shardд
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameж
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*ч
valueыBЬB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesћ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B B B B 2
SaveV2/shape_and_slicesі
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0)savev2_layer_1_kernel_read_readvariableop'savev2_layer_1_bias_read_readvariableop(savev2_output_kernel_read_readvariableop&savev2_output_bias_read_readvariableop#savev2_variable_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes

22
SaveV2║
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesА
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
п
|
'__inference_output_layer_call_fn_633337

inputs
unknown
	unknown_0
identityѕбStatefulPartitionedCallЫ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *K
fFRD
B__inference_output_layer_call_and_return_conditional_losses_6325162
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*.
_input_shapes
:         2::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         2
 
_user_specified_nameinputs
╩
С
"__inference__traced_restore_633400
file_prefix#
assignvariableop_layer_1_kernel#
assignvariableop_1_layer_1_bias$
 assignvariableop_2_output_kernel"
assignvariableop_3_output_bias
assignvariableop_4_variable

identity_6ѕбAssignVariableOpбAssignVariableOp_1бAssignVariableOp_2бAssignVariableOp_3бAssignVariableOp_4№
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*ч
valueыBЬB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesџ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B B B B 2
RestoreV2/shape_and_slices╔
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

Identityъ
AssignVariableOpAssignVariableOpassignvariableop_layer_1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1ц
AssignVariableOp_1AssignVariableOpassignvariableop_1_layer_1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2Ц
AssignVariableOp_2AssignVariableOp assignvariableop_2_output_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3Б
AssignVariableOp_3AssignVariableOpassignvariableop_3_output_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4а
AssignVariableOp_4AssignVariableOpassignvariableop_4_variableIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp¤

Identity_5Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_5┴

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
┴
Г
+__inference_sequential_layer_call_fn_633215

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identityѕбStatefulPartitionedCallЮ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_6326022
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*:
_input_shapes)
':         	:::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         	
 
_user_specified_nameinputs
»
ф
B__inference_output_layer_call_and_return_conditional_losses_633328

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityѕЇ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:2*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpЂ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:         2	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*.
_input_shapes
:         2:::O K
'
_output_shapes
:         2
 
_user_specified_nameinputs
е
Ф
C__inference_layer-1_layer_call_and_return_conditional_losses_632489

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityѕЇ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:	2*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         22
MatMulї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype02
BiasAdd/ReadVariableOpЂ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         22	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:         22
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:         22

Identity"
identityIdentity:output:0*.
_input_shapes
:         	:::O K
'
_output_shapes
:         	
 
_user_specified_nameinputs
Ф
i
(__inference_project_layer_call_fn_632471
x
unknown
identityѕбStatefulPartitionedCallр
StatefulPartitionedCallStatefulPartitionedCallxunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         	*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_project_layer_call_and_return_conditional_losses_6324632
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         	2

Identity"
identityIdentity:output:0**
_input_shapes
:         	:22
StatefulPartitionedCallStatefulPartitionedCall:J F
'
_output_shapes
:         	

_user_specified_namex
е
Ф
C__inference_layer-1_layer_call_and_return_conditional_losses_633308

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityѕЇ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:	2*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         22
MatMulї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype02
BiasAdd/ReadVariableOpЂ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         22	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:         22
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:         22

Identity"
identityIdentity:output:0*.
_input_shapes
:         	:::O K
'
_output_shapes
:         	
 
_user_specified_nameinputs
┼
»
-__inference_functional_1_layer_call_fn_632969

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identityѕбStatefulPartitionedCallЪ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8ѓ *Q
fLRJ
H__inference_functional_1_layer_call_and_return_conditional_losses_6328572
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*:
_input_shapes)
':         	:::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         	
 
_user_specified_nameinputs
Ь	
ц
H__inference_functional_1_layer_call_and_return_conditional_losses_632809
input_1
classifier_graph_632797
classifier_graph_632799
classifier_graph_632801
classifier_graph_632803
classifier_graph_632805
identityѕб(classifier_graph/StatefulPartitionedCallј
(classifier_graph/StatefulPartitionedCallStatefulPartitionedCallinput_1classifier_graph_632797classifier_graph_632799classifier_graph_632801classifier_graph_632803classifier_graph_632805*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8ѓ *U
fPRN
L__inference_classifier_graph_layer_call_and_return_conditional_losses_6326942*
(classifier_graph/StatefulPartitionedCall░
IdentityIdentity1classifier_graph/StatefulPartitionedCall:output:0)^classifier_graph/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*:
_input_shapes)
':         	:::::2T
(classifier_graph/StatefulPartitionedCall(classifier_graph/StatefulPartitionedCall:P L
'
_output_shapes
:         	
!
_user_specified_name	input_1
»
ф
B__inference_output_layer_call_and_return_conditional_losses_632516

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityѕЇ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:2*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpЂ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:         2	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*.
_input_shapes
:         2:::O K
'
_output_shapes
:         2
 
_user_specified_nameinputs
┤
м
F__inference_sequential_layer_call_and_return_conditional_losses_633267
project_input4
0project_matrix_transpose_readvariableop_resource*
&layer_1_matmul_readvariableop_resource+
'layer_1_biasadd_readvariableop_resource)
%output_matmul_readvariableop_resource*
&output_biasadd_readvariableop_resource
identityѕ├
'project/matrix_transpose/ReadVariableOpReadVariableOp0project_matrix_transpose_readvariableop_resource*
_output_shapes

:	*
dtype02)
'project/matrix_transpose/ReadVariableOpБ
'project/matrix_transpose/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2)
'project/matrix_transpose/transpose/permр
"project/matrix_transpose/transpose	Transpose/project/matrix_transpose/ReadVariableOp:value:00project/matrix_transpose/transpose/perm:output:0*
T0*
_output_shapes

:	2$
"project/matrix_transpose/transposeЊ
project/matmulMatMulproject_input&project/matrix_transpose/transpose:y:0*
T0*'
_output_shapes
:         2
project/matmul│
project/matmul_1/ReadVariableOpReadVariableOp0project_matrix_transpose_readvariableop_resource*
_output_shapes

:	*
dtype02!
project/matmul_1/ReadVariableOpБ
project/matmul_1MatMulproject/matmul:product:0'project/matmul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         	2
project/matmul_1~
project/subSubproject_inputproject/matmul_1:product:0*
T0*'
_output_shapes
:         	2
project/subЦ
layer-1/MatMul/ReadVariableOpReadVariableOp&layer_1_matmul_readvariableop_resource*
_output_shapes

:	2*
dtype02
layer-1/MatMul/ReadVariableOpћ
layer-1/MatMulMatMulproject/sub:z:0%layer-1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         22
layer-1/MatMulц
layer-1/BiasAdd/ReadVariableOpReadVariableOp'layer_1_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02 
layer-1/BiasAdd/ReadVariableOpА
layer-1/BiasAddBiasAddlayer-1/MatMul:product:0&layer-1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         22
layer-1/BiasAddp
layer-1/ReluRelulayer-1/BiasAdd:output:0*
T0*'
_output_shapes
:         22
layer-1/Reluб
output/MatMul/ReadVariableOpReadVariableOp%output_matmul_readvariableop_resource*
_output_shapes

:2*
dtype02
output/MatMul/ReadVariableOpю
output/MatMulMatMullayer-1/Relu:activations:0$output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
output/MatMulА
output/BiasAdd/ReadVariableOpReadVariableOp&output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
output/BiasAdd/ReadVariableOpЮ
output/BiasAddBiasAddoutput/MatMul:product:0%output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
output/BiasAddv
output/SoftmaxSoftmaxoutput/BiasAdd:output:0*
T0*'
_output_shapes
:         2
output/Softmaxl
IdentityIdentityoutput/Softmax:softmax:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*:
_input_shapes)
':         	::::::V R
'
_output_shapes
:         	
'
_user_specified_nameproject_input
Т'
┘
H__inference_functional_1_layer_call_and_return_conditional_losses_632939

inputsP
Lclassifier_graph_sequential_project_matrix_transpose_readvariableop_resourceF
Bclassifier_graph_sequential_layer_1_matmul_readvariableop_resourceG
Cclassifier_graph_sequential_layer_1_biasadd_readvariableop_resourceE
Aclassifier_graph_sequential_output_matmul_readvariableop_resourceF
Bclassifier_graph_sequential_output_biasadd_readvariableop_resource
identityѕЌ
Cclassifier_graph/sequential/project/matrix_transpose/ReadVariableOpReadVariableOpLclassifier_graph_sequential_project_matrix_transpose_readvariableop_resource*
_output_shapes

:	*
dtype02E
Cclassifier_graph/sequential/project/matrix_transpose/ReadVariableOp█
Cclassifier_graph/sequential/project/matrix_transpose/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2E
Cclassifier_graph/sequential/project/matrix_transpose/transpose/permЛ
>classifier_graph/sequential/project/matrix_transpose/transpose	TransposeKclassifier_graph/sequential/project/matrix_transpose/ReadVariableOp:value:0Lclassifier_graph/sequential/project/matrix_transpose/transpose/perm:output:0*
T0*
_output_shapes

:	2@
>classifier_graph/sequential/project/matrix_transpose/transposeЯ
*classifier_graph/sequential/project/matmulMatMulinputsBclassifier_graph/sequential/project/matrix_transpose/transpose:y:0*
T0*'
_output_shapes
:         2,
*classifier_graph/sequential/project/matmulЄ
;classifier_graph/sequential/project/matmul_1/ReadVariableOpReadVariableOpLclassifier_graph_sequential_project_matrix_transpose_readvariableop_resource*
_output_shapes

:	*
dtype02=
;classifier_graph/sequential/project/matmul_1/ReadVariableOpЊ
,classifier_graph/sequential/project/matmul_1MatMul4classifier_graph/sequential/project/matmul:product:0Cclassifier_graph/sequential/project/matmul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         	2.
,classifier_graph/sequential/project/matmul_1╦
'classifier_graph/sequential/project/subSubinputs6classifier_graph/sequential/project/matmul_1:product:0*
T0*'
_output_shapes
:         	2)
'classifier_graph/sequential/project/subщ
9classifier_graph/sequential/layer-1/MatMul/ReadVariableOpReadVariableOpBclassifier_graph_sequential_layer_1_matmul_readvariableop_resource*
_output_shapes

:	2*
dtype02;
9classifier_graph/sequential/layer-1/MatMul/ReadVariableOpё
*classifier_graph/sequential/layer-1/MatMulMatMul+classifier_graph/sequential/project/sub:z:0Aclassifier_graph/sequential/layer-1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         22,
*classifier_graph/sequential/layer-1/MatMulЭ
:classifier_graph/sequential/layer-1/BiasAdd/ReadVariableOpReadVariableOpCclassifier_graph_sequential_layer_1_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02<
:classifier_graph/sequential/layer-1/BiasAdd/ReadVariableOpЉ
+classifier_graph/sequential/layer-1/BiasAddBiasAdd4classifier_graph/sequential/layer-1/MatMul:product:0Bclassifier_graph/sequential/layer-1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         22-
+classifier_graph/sequential/layer-1/BiasAdd─
(classifier_graph/sequential/layer-1/ReluRelu4classifier_graph/sequential/layer-1/BiasAdd:output:0*
T0*'
_output_shapes
:         22*
(classifier_graph/sequential/layer-1/ReluШ
8classifier_graph/sequential/output/MatMul/ReadVariableOpReadVariableOpAclassifier_graph_sequential_output_matmul_readvariableop_resource*
_output_shapes

:2*
dtype02:
8classifier_graph/sequential/output/MatMul/ReadVariableOpї
)classifier_graph/sequential/output/MatMulMatMul6classifier_graph/sequential/layer-1/Relu:activations:0@classifier_graph/sequential/output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2+
)classifier_graph/sequential/output/MatMulш
9classifier_graph/sequential/output/BiasAdd/ReadVariableOpReadVariableOpBclassifier_graph_sequential_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02;
9classifier_graph/sequential/output/BiasAdd/ReadVariableOpЇ
*classifier_graph/sequential/output/BiasAddBiasAdd3classifier_graph/sequential/output/MatMul:product:0Aclassifier_graph/sequential/output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2,
*classifier_graph/sequential/output/BiasAdd╩
*classifier_graph/sequential/output/SoftmaxSoftmax3classifier_graph/sequential/output/BiasAdd:output:0*
T0*'
_output_shapes
:         2,
*classifier_graph/sequential/output/Softmaxѕ
IdentityIdentity4classifier_graph/sequential/output/Softmax:softmax:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*:
_input_shapes)
':         	::::::O K
'
_output_shapes
:         	
 
_user_specified_nameinputs
┤
м
F__inference_sequential_layer_call_and_return_conditional_losses_633241
project_input4
0project_matrix_transpose_readvariableop_resource*
&layer_1_matmul_readvariableop_resource+
'layer_1_biasadd_readvariableop_resource)
%output_matmul_readvariableop_resource*
&output_biasadd_readvariableop_resource
identityѕ├
'project/matrix_transpose/ReadVariableOpReadVariableOp0project_matrix_transpose_readvariableop_resource*
_output_shapes

:	*
dtype02)
'project/matrix_transpose/ReadVariableOpБ
'project/matrix_transpose/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2)
'project/matrix_transpose/transpose/permр
"project/matrix_transpose/transpose	Transpose/project/matrix_transpose/ReadVariableOp:value:00project/matrix_transpose/transpose/perm:output:0*
T0*
_output_shapes

:	2$
"project/matrix_transpose/transposeЊ
project/matmulMatMulproject_input&project/matrix_transpose/transpose:y:0*
T0*'
_output_shapes
:         2
project/matmul│
project/matmul_1/ReadVariableOpReadVariableOp0project_matrix_transpose_readvariableop_resource*
_output_shapes

:	*
dtype02!
project/matmul_1/ReadVariableOpБ
project/matmul_1MatMulproject/matmul:product:0'project/matmul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         	2
project/matmul_1~
project/subSubproject_inputproject/matmul_1:product:0*
T0*'
_output_shapes
:         	2
project/subЦ
layer-1/MatMul/ReadVariableOpReadVariableOp&layer_1_matmul_readvariableop_resource*
_output_shapes

:	2*
dtype02
layer-1/MatMul/ReadVariableOpћ
layer-1/MatMulMatMulproject/sub:z:0%layer-1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         22
layer-1/MatMulц
layer-1/BiasAdd/ReadVariableOpReadVariableOp'layer_1_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02 
layer-1/BiasAdd/ReadVariableOpА
layer-1/BiasAddBiasAddlayer-1/MatMul:product:0&layer-1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         22
layer-1/BiasAddp
layer-1/ReluRelulayer-1/BiasAdd:output:0*
T0*'
_output_shapes
:         22
layer-1/Reluб
output/MatMul/ReadVariableOpReadVariableOp%output_matmul_readvariableop_resource*
_output_shapes

:2*
dtype02
output/MatMul/ReadVariableOpю
output/MatMulMatMullayer-1/Relu:activations:0$output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
output/MatMulА
output/BiasAdd/ReadVariableOpReadVariableOp&output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
output/BiasAdd/ReadVariableOpЮ
output/BiasAddBiasAddoutput/MatMul:product:0%output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
output/BiasAddv
output/SoftmaxSoftmaxoutput/BiasAdd:output:0*
T0*'
_output_shapes
:         2
output/Softmaxl
IdentityIdentityoutput/Softmax:softmax:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*:
_input_shapes)
':         	::::::V R
'
_output_shapes
:         	
'
_user_specified_nameproject_input
ж-
З
!__inference__wrapped_model_632450
input_1]
Yfunctional_1_classifier_graph_sequential_project_matrix_transpose_readvariableop_resourceS
Ofunctional_1_classifier_graph_sequential_layer_1_matmul_readvariableop_resourceT
Pfunctional_1_classifier_graph_sequential_layer_1_biasadd_readvariableop_resourceR
Nfunctional_1_classifier_graph_sequential_output_matmul_readvariableop_resourceS
Ofunctional_1_classifier_graph_sequential_output_biasadd_readvariableop_resource
identityѕЙ
Pfunctional_1/classifier_graph/sequential/project/matrix_transpose/ReadVariableOpReadVariableOpYfunctional_1_classifier_graph_sequential_project_matrix_transpose_readvariableop_resource*
_output_shapes

:	*
dtype02R
Pfunctional_1/classifier_graph/sequential/project/matrix_transpose/ReadVariableOpш
Pfunctional_1/classifier_graph/sequential/project/matrix_transpose/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2R
Pfunctional_1/classifier_graph/sequential/project/matrix_transpose/transpose/permЁ
Kfunctional_1/classifier_graph/sequential/project/matrix_transpose/transpose	TransposeXfunctional_1/classifier_graph/sequential/project/matrix_transpose/ReadVariableOp:value:0Yfunctional_1/classifier_graph/sequential/project/matrix_transpose/transpose/perm:output:0*
T0*
_output_shapes

:	2M
Kfunctional_1/classifier_graph/sequential/project/matrix_transpose/transposeѕ
7functional_1/classifier_graph/sequential/project/matmulMatMulinput_1Ofunctional_1/classifier_graph/sequential/project/matrix_transpose/transpose:y:0*
T0*'
_output_shapes
:         29
7functional_1/classifier_graph/sequential/project/matmul«
Hfunctional_1/classifier_graph/sequential/project/matmul_1/ReadVariableOpReadVariableOpYfunctional_1_classifier_graph_sequential_project_matrix_transpose_readvariableop_resource*
_output_shapes

:	*
dtype02J
Hfunctional_1/classifier_graph/sequential/project/matmul_1/ReadVariableOpК
9functional_1/classifier_graph/sequential/project/matmul_1MatMulAfunctional_1/classifier_graph/sequential/project/matmul:product:0Pfunctional_1/classifier_graph/sequential/project/matmul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         	2;
9functional_1/classifier_graph/sequential/project/matmul_1з
4functional_1/classifier_graph/sequential/project/subSubinput_1Cfunctional_1/classifier_graph/sequential/project/matmul_1:product:0*
T0*'
_output_shapes
:         	26
4functional_1/classifier_graph/sequential/project/subа
Ffunctional_1/classifier_graph/sequential/layer-1/MatMul/ReadVariableOpReadVariableOpOfunctional_1_classifier_graph_sequential_layer_1_matmul_readvariableop_resource*
_output_shapes

:	2*
dtype02H
Ffunctional_1/classifier_graph/sequential/layer-1/MatMul/ReadVariableOpИ
7functional_1/classifier_graph/sequential/layer-1/MatMulMatMul8functional_1/classifier_graph/sequential/project/sub:z:0Nfunctional_1/classifier_graph/sequential/layer-1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         229
7functional_1/classifier_graph/sequential/layer-1/MatMulЪ
Gfunctional_1/classifier_graph/sequential/layer-1/BiasAdd/ReadVariableOpReadVariableOpPfunctional_1_classifier_graph_sequential_layer_1_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02I
Gfunctional_1/classifier_graph/sequential/layer-1/BiasAdd/ReadVariableOp┼
8functional_1/classifier_graph/sequential/layer-1/BiasAddBiasAddAfunctional_1/classifier_graph/sequential/layer-1/MatMul:product:0Ofunctional_1/classifier_graph/sequential/layer-1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         22:
8functional_1/classifier_graph/sequential/layer-1/BiasAddв
5functional_1/classifier_graph/sequential/layer-1/ReluReluAfunctional_1/classifier_graph/sequential/layer-1/BiasAdd:output:0*
T0*'
_output_shapes
:         227
5functional_1/classifier_graph/sequential/layer-1/ReluЮ
Efunctional_1/classifier_graph/sequential/output/MatMul/ReadVariableOpReadVariableOpNfunctional_1_classifier_graph_sequential_output_matmul_readvariableop_resource*
_output_shapes

:2*
dtype02G
Efunctional_1/classifier_graph/sequential/output/MatMul/ReadVariableOp└
6functional_1/classifier_graph/sequential/output/MatMulMatMulCfunctional_1/classifier_graph/sequential/layer-1/Relu:activations:0Mfunctional_1/classifier_graph/sequential/output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         28
6functional_1/classifier_graph/sequential/output/MatMulю
Ffunctional_1/classifier_graph/sequential/output/BiasAdd/ReadVariableOpReadVariableOpOfunctional_1_classifier_graph_sequential_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02H
Ffunctional_1/classifier_graph/sequential/output/BiasAdd/ReadVariableOp┴
7functional_1/classifier_graph/sequential/output/BiasAddBiasAdd@functional_1/classifier_graph/sequential/output/MatMul:product:0Nfunctional_1/classifier_graph/sequential/output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         29
7functional_1/classifier_graph/sequential/output/BiasAddы
7functional_1/classifier_graph/sequential/output/SoftmaxSoftmax@functional_1/classifier_graph/sequential/output/BiasAdd:output:0*
T0*'
_output_shapes
:         29
7functional_1/classifier_graph/sequential/output/SoftmaxЋ
IdentityIdentityAfunctional_1/classifier_graph/sequential/output/Softmax:softmax:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*:
_input_shapes)
':         	::::::P L
'
_output_shapes
:         	
!
_user_specified_name	input_1
Т'
┘
H__inference_functional_1_layer_call_and_return_conditional_losses_632913

inputsP
Lclassifier_graph_sequential_project_matrix_transpose_readvariableop_resourceF
Bclassifier_graph_sequential_layer_1_matmul_readvariableop_resourceG
Cclassifier_graph_sequential_layer_1_biasadd_readvariableop_resourceE
Aclassifier_graph_sequential_output_matmul_readvariableop_resourceF
Bclassifier_graph_sequential_output_biasadd_readvariableop_resource
identityѕЌ
Cclassifier_graph/sequential/project/matrix_transpose/ReadVariableOpReadVariableOpLclassifier_graph_sequential_project_matrix_transpose_readvariableop_resource*
_output_shapes

:	*
dtype02E
Cclassifier_graph/sequential/project/matrix_transpose/ReadVariableOp█
Cclassifier_graph/sequential/project/matrix_transpose/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2E
Cclassifier_graph/sequential/project/matrix_transpose/transpose/permЛ
>classifier_graph/sequential/project/matrix_transpose/transpose	TransposeKclassifier_graph/sequential/project/matrix_transpose/ReadVariableOp:value:0Lclassifier_graph/sequential/project/matrix_transpose/transpose/perm:output:0*
T0*
_output_shapes

:	2@
>classifier_graph/sequential/project/matrix_transpose/transposeЯ
*classifier_graph/sequential/project/matmulMatMulinputsBclassifier_graph/sequential/project/matrix_transpose/transpose:y:0*
T0*'
_output_shapes
:         2,
*classifier_graph/sequential/project/matmulЄ
;classifier_graph/sequential/project/matmul_1/ReadVariableOpReadVariableOpLclassifier_graph_sequential_project_matrix_transpose_readvariableop_resource*
_output_shapes

:	*
dtype02=
;classifier_graph/sequential/project/matmul_1/ReadVariableOpЊ
,classifier_graph/sequential/project/matmul_1MatMul4classifier_graph/sequential/project/matmul:product:0Cclassifier_graph/sequential/project/matmul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         	2.
,classifier_graph/sequential/project/matmul_1╦
'classifier_graph/sequential/project/subSubinputs6classifier_graph/sequential/project/matmul_1:product:0*
T0*'
_output_shapes
:         	2)
'classifier_graph/sequential/project/subщ
9classifier_graph/sequential/layer-1/MatMul/ReadVariableOpReadVariableOpBclassifier_graph_sequential_layer_1_matmul_readvariableop_resource*
_output_shapes

:	2*
dtype02;
9classifier_graph/sequential/layer-1/MatMul/ReadVariableOpё
*classifier_graph/sequential/layer-1/MatMulMatMul+classifier_graph/sequential/project/sub:z:0Aclassifier_graph/sequential/layer-1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         22,
*classifier_graph/sequential/layer-1/MatMulЭ
:classifier_graph/sequential/layer-1/BiasAdd/ReadVariableOpReadVariableOpCclassifier_graph_sequential_layer_1_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02<
:classifier_graph/sequential/layer-1/BiasAdd/ReadVariableOpЉ
+classifier_graph/sequential/layer-1/BiasAddBiasAdd4classifier_graph/sequential/layer-1/MatMul:product:0Bclassifier_graph/sequential/layer-1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         22-
+classifier_graph/sequential/layer-1/BiasAdd─
(classifier_graph/sequential/layer-1/ReluRelu4classifier_graph/sequential/layer-1/BiasAdd:output:0*
T0*'
_output_shapes
:         22*
(classifier_graph/sequential/layer-1/ReluШ
8classifier_graph/sequential/output/MatMul/ReadVariableOpReadVariableOpAclassifier_graph_sequential_output_matmul_readvariableop_resource*
_output_shapes

:2*
dtype02:
8classifier_graph/sequential/output/MatMul/ReadVariableOpї
)classifier_graph/sequential/output/MatMulMatMul6classifier_graph/sequential/layer-1/Relu:activations:0@classifier_graph/sequential/output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2+
)classifier_graph/sequential/output/MatMulш
9classifier_graph/sequential/output/BiasAdd/ReadVariableOpReadVariableOpBclassifier_graph_sequential_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02;
9classifier_graph/sequential/output/BiasAdd/ReadVariableOpЇ
*classifier_graph/sequential/output/BiasAddBiasAdd3classifier_graph/sequential/output/MatMul:product:0Aclassifier_graph/sequential/output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2,
*classifier_graph/sequential/output/BiasAdd╩
*classifier_graph/sequential/output/SoftmaxSoftmax3classifier_graph/sequential/output/BiasAdd:output:0*
T0*'
_output_shapes
:         2,
*classifier_graph/sequential/output/Softmaxѕ
IdentityIdentity4classifier_graph/sequential/output/Softmax:softmax:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*:
_input_shapes)
':         	::::::O K
'
_output_shapes
:         	
 
_user_specified_nameinputs
ч

І
C__inference_project_layer_call_and_return_conditional_losses_632463
x,
(matrix_transpose_readvariableop_resource
identityѕФ
matrix_transpose/ReadVariableOpReadVariableOp(matrix_transpose_readvariableop_resource*
_output_shapes

:	*
dtype02!
matrix_transpose/ReadVariableOpЊ
matrix_transpose/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2!
matrix_transpose/transpose/perm┴
matrix_transpose/transpose	Transpose'matrix_transpose/ReadVariableOp:value:0(matrix_transpose/transpose/perm:output:0*
T0*
_output_shapes

:	2
matrix_transpose/transposeo
matmulMatMulxmatrix_transpose/transpose:y:0*
T0*'
_output_shapes
:         2
matmulЏ
matmul_1/ReadVariableOpReadVariableOp(matrix_transpose_readvariableop_resource*
_output_shapes

:	*
dtype02
matmul_1/ReadVariableOpЃ
matmul_1MatMulmatmul:product:0matmul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         	2

matmul_1Z
subSubxmatmul_1:product:0*
T0*'
_output_shapes
:         	2
sub[
IdentityIdentitysub:z:0*
T0*'
_output_shapes
:         	2

Identity"
identityIdentity:output:0**
_input_shapes
:         	::J F
'
_output_shapes
:         	

_user_specified_namex
в	
Б
H__inference_functional_1_layer_call_and_return_conditional_losses_632857

inputs
classifier_graph_632845
classifier_graph_632847
classifier_graph_632849
classifier_graph_632851
classifier_graph_632853
identityѕб(classifier_graph/StatefulPartitionedCallЇ
(classifier_graph/StatefulPartitionedCallStatefulPartitionedCallinputsclassifier_graph_632845classifier_graph_632847classifier_graph_632849classifier_graph_632851classifier_graph_632853*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8ѓ *U
fPRN
L__inference_classifier_graph_layer_call_and_return_conditional_losses_6326942*
(classifier_graph/StatefulPartitionedCall░
IdentityIdentity1classifier_graph/StatefulPartitionedCall:output:0)^classifier_graph/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*:
_input_shapes)
':         	:::::2T
(classifier_graph/StatefulPartitionedCall(classifier_graph/StatefulPartitionedCall:O K
'
_output_shapes
:         	
 
_user_specified_nameinputs
о
┤
+__inference_sequential_layer_call_fn_633297
project_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identityѕбStatefulPartitionedCallц
StatefulPartitionedCallStatefulPartitionedCallproject_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_6326022
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*:
_input_shapes)
':         	:::::22
StatefulPartitionedCallStatefulPartitionedCall:V R
'
_output_shapes
:         	
'
_user_specified_nameproject_input
┴
Г
+__inference_sequential_layer_call_fn_633200

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identityѕбStatefulPartitionedCallЮ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_6325702
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*:
_input_shapes)
':         	:::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         	
 
_user_specified_nameinputs
Ь	
ц
H__inference_functional_1_layer_call_and_return_conditional_losses_632794
input_1
classifier_graph_632782
classifier_graph_632784
classifier_graph_632786
classifier_graph_632788
classifier_graph_632790
identityѕб(classifier_graph/StatefulPartitionedCallј
(classifier_graph/StatefulPartitionedCallStatefulPartitionedCallinput_1classifier_graph_632782classifier_graph_632784classifier_graph_632786classifier_graph_632788classifier_graph_632790*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8ѓ *U
fPRN
L__inference_classifier_graph_layer_call_and_return_conditional_losses_6327512*
(classifier_graph/StatefulPartitionedCall░
IdentityIdentity1classifier_graph/StatefulPartitionedCall:output:0)^classifier_graph/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*:
_input_shapes)
':         	:::::2T
(classifier_graph/StatefulPartitionedCall(classifier_graph/StatefulPartitionedCall:P L
'
_output_shapes
:         	
!
_user_specified_name	input_1
╚
░
-__inference_functional_1_layer_call_fn_632870
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identityѕбStatefulPartitionedCallа
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8ѓ *Q
fLRJ
H__inference_functional_1_layer_call_and_return_conditional_losses_6328572
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*:
_input_shapes)
':         	:::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:         	
!
_user_specified_name	input_1
╚
░
-__inference_functional_1_layer_call_fn_632840
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identityѕбStatefulPartitionedCallа
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8ѓ *Q
fLRJ
H__inference_functional_1_layer_call_and_return_conditional_losses_6328272
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*:
_input_shapes)
':         	:::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:         	
!
_user_specified_name	input_1
и
Ѓ
L__inference_classifier_graph_layer_call_and_return_conditional_losses_633077
x?
;sequential_project_matrix_transpose_readvariableop_resource5
1sequential_layer_1_matmul_readvariableop_resource6
2sequential_layer_1_biasadd_readvariableop_resource4
0sequential_output_matmul_readvariableop_resource5
1sequential_output_biasadd_readvariableop_resource
identityѕС
2sequential/project/matrix_transpose/ReadVariableOpReadVariableOp;sequential_project_matrix_transpose_readvariableop_resource*
_output_shapes

:	*
dtype024
2sequential/project/matrix_transpose/ReadVariableOp╣
2sequential/project/matrix_transpose/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       24
2sequential/project/matrix_transpose/transpose/permЇ
-sequential/project/matrix_transpose/transpose	Transpose:sequential/project/matrix_transpose/ReadVariableOp:value:0;sequential/project/matrix_transpose/transpose/perm:output:0*
T0*
_output_shapes

:	2/
-sequential/project/matrix_transpose/transposeе
sequential/project/matmulMatMulx1sequential/project/matrix_transpose/transpose:y:0*
T0*'
_output_shapes
:         2
sequential/project/matmulн
*sequential/project/matmul_1/ReadVariableOpReadVariableOp;sequential_project_matrix_transpose_readvariableop_resource*
_output_shapes

:	*
dtype02,
*sequential/project/matmul_1/ReadVariableOp¤
sequential/project/matmul_1MatMul#sequential/project/matmul:product:02sequential/project/matmul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         	2
sequential/project/matmul_1Њ
sequential/project/subSubx%sequential/project/matmul_1:product:0*
T0*'
_output_shapes
:         	2
sequential/project/subк
(sequential/layer-1/MatMul/ReadVariableOpReadVariableOp1sequential_layer_1_matmul_readvariableop_resource*
_output_shapes

:	2*
dtype02*
(sequential/layer-1/MatMul/ReadVariableOp└
sequential/layer-1/MatMulMatMulsequential/project/sub:z:00sequential/layer-1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         22
sequential/layer-1/MatMul┼
)sequential/layer-1/BiasAdd/ReadVariableOpReadVariableOp2sequential_layer_1_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02+
)sequential/layer-1/BiasAdd/ReadVariableOp═
sequential/layer-1/BiasAddBiasAdd#sequential/layer-1/MatMul:product:01sequential/layer-1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         22
sequential/layer-1/BiasAddЉ
sequential/layer-1/ReluRelu#sequential/layer-1/BiasAdd:output:0*
T0*'
_output_shapes
:         22
sequential/layer-1/Relu├
'sequential/output/MatMul/ReadVariableOpReadVariableOp0sequential_output_matmul_readvariableop_resource*
_output_shapes

:2*
dtype02)
'sequential/output/MatMul/ReadVariableOp╚
sequential/output/MatMulMatMul%sequential/layer-1/Relu:activations:0/sequential/output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
sequential/output/MatMul┬
(sequential/output/BiasAdd/ReadVariableOpReadVariableOp1sequential_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(sequential/output/BiasAdd/ReadVariableOp╔
sequential/output/BiasAddBiasAdd"sequential/output/MatMul:product:00sequential/output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
sequential/output/BiasAddЌ
sequential/output/SoftmaxSoftmax"sequential/output/BiasAdd:output:0*
T0*'
_output_shapes
:         2
sequential/output/Softmaxw
IdentityIdentity#sequential/output/Softmax:softmax:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*:
_input_shapes)
':         	::::::J F
'
_output_shapes
:         	

_user_specified_namex
┼
»
-__inference_functional_1_layer_call_fn_632954

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identityѕбStatefulPartitionedCallЪ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8ѓ *Q
fLRJ
H__inference_functional_1_layer_call_and_return_conditional_losses_6328272
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*:
_input_shapes)
':         	:::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         	
 
_user_specified_nameinputs
в
г
F__inference_sequential_layer_call_and_return_conditional_losses_632570

inputs
project_632556
layer_1_632559
layer_1_632561
output_632564
output_632566
identityѕбlayer-1/StatefulPartitionedCallбoutput/StatefulPartitionedCallбproject/StatefulPartitionedCall§
project/StatefulPartitionedCallStatefulPartitionedCallinputsproject_632556*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         	*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_project_layer_call_and_return_conditional_losses_6324632!
project/StatefulPartitionedCall▒
layer-1/StatefulPartitionedCallStatefulPartitionedCall(project/StatefulPartitionedCall:output:0layer_1_632559layer_1_632561*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_layer-1_layer_call_and_return_conditional_losses_6324892!
layer-1/StatefulPartitionedCallг
output/StatefulPartitionedCallStatefulPartitionedCall(layer-1/StatefulPartitionedCall:output:0output_632564output_632566*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *K
fFRD
B__inference_output_layer_call_and_return_conditional_losses_6325162 
output/StatefulPartitionedCallЯ
IdentityIdentity'output/StatefulPartitionedCall:output:0 ^layer-1/StatefulPartitionedCall^output/StatefulPartitionedCall ^project/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*:
_input_shapes)
':         	:::::2B
layer-1/StatefulPartitionedCalllayer-1/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall2B
project/StatefulPartitionedCallproject/StatefulPartitionedCall:O K
'
_output_shapes
:         	
 
_user_specified_nameinputs"ИL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*│
serving_defaultЪ
;
input_10
serving_default_input_1:0         	D
classifier_graph0
StatefulPartitionedCall:0         tensorflow/serving/predict:Ћб
і	
layer-0
layer_with_weights-0
layer-1
regularization_losses
	variables
trainable_variables
	keras_api

signatures
*I&call_and_return_all_conditional_losses
J_default_save_signature
K__call__"џ
_tf_keras_network■{"class_name": "Functional", "name": "functional_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "functional_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 9]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "ClassifierGraph", "config": {"layer was saved without config": true}, "name": "classifier_graph", "inbound_nodes": [[["input_1", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["classifier_graph", 0, 0]]}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 9]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional"}}
ж"Т
_tf_keras_input_layerк{"class_name": "InputLayer", "name": "input_1", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 9]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 9]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}}
Х

Layers
		model

regularization_losses
	variables
trainable_variables
	keras_api
*L&call_and_return_all_conditional_losses
M__call__"љ
_tf_keras_modelШ{"class_name": "ClassifierGraph", "name": "classifier_graph", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "is_graph_network": false, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "ClassifierGraph"}}
 "
trackable_list_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
╩
regularization_losses
layer_regularization_losses
non_trainable_variables
	variables
metrics
layer_metrics
trainable_variables

layers
K__call__
J_default_save_signature
*I&call_and_return_all_conditional_losses
&I"call_and_return_conditional_losses"
_generic_user_object
,
Nserving_default"
signature_map
5
0
1
2"
trackable_list_wrapper
Е
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
regularization_losses
	variables
trainable_variables
	keras_api
*O&call_and_return_all_conditional_losses
P__call__"Ц
_tf_keras_sequentialє{"class_name": "Sequential", "name": "sequential", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 9]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "project_input"}}, {"class_name": "Project", "config": {"layer was saved without config": true}}, {"class_name": "Dense", "config": {"name": "layer-1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 9]}, "dtype": "float32", "units": 50, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "output", "trainable": true, "dtype": "float32", "units": 2, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 9]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 9]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential"}}
 "
trackable_list_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
Г

regularization_losses
layer_regularization_losses
 non_trainable_variables
	variables
!metrics
"layer_metrics
trainable_variables

#layers
M__call__
*L&call_and_return_all_conditional_losses
&L"call_and_return_conditional_losses"
_generic_user_object
 :	22layer-1/kernel
:22layer-1/bias
:22output/kernel
:2output/bias
:	2Variable
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
0
1"
trackable_list_wrapper
І
w
$_inbound_nodes
%_outbound_nodes
&regularization_losses
'	variables
(trainable_variables
)	keras_api
*Q&call_and_return_all_conditional_losses
R__call__"╠
_tf_keras_layer▓{"class_name": "Project", "name": "project", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 9]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}}
Є	
*_inbound_nodes

kernel
bias
+_outbound_nodes
,regularization_losses
-	variables
.trainable_variables
/	keras_api
*S&call_and_return_all_conditional_losses
T__call__"╣
_tf_keras_layerЪ{"class_name": "Dense", "name": "layer-1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 9]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "layer-1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 9]}, "dtype": "float32", "units": 50, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 9}}}, "build_input_shape": {"class_name": "TensorShape", "items": [250, 9]}}
Ѓ
0_inbound_nodes

kernel
bias
1regularization_losses
2	variables
3trainable_variables
4	keras_api
*U&call_and_return_all_conditional_losses
V__call__"╩
_tf_keras_layer░{"class_name": "Dense", "name": "output", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "output", "trainable": true, "dtype": "float32", "units": 2, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 50}}}, "build_input_shape": {"class_name": "TensorShape", "items": [250, 50]}}
 "
trackable_list_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
Г
regularization_losses
5layer_regularization_losses
6non_trainable_variables
	variables
7metrics
8layer_metrics
trainable_variables

9layers
P__call__
*O&call_and_return_all_conditional_losses
&O"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
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
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
Г
&regularization_losses
:layer_regularization_losses
;non_trainable_variables
'	variables
<metrics
=layer_metrics
(trainable_variables

>layers
R__call__
*Q&call_and_return_all_conditional_losses
&Q"call_and_return_conditional_losses"
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
Г
,regularization_losses
?layer_regularization_losses
@non_trainable_variables
-	variables
Ametrics
Blayer_metrics
.trainable_variables

Clayers
T__call__
*S&call_and_return_all_conditional_losses
&S"call_and_return_conditional_losses"
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
Г
1regularization_losses
Dlayer_regularization_losses
Enon_trainable_variables
2	variables
Fmetrics
Glayer_metrics
3trainable_variables

Hlayers
V__call__
*U&call_and_return_all_conditional_losses
&U"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
5
0
1
2"
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
 "
trackable_list_wrapper
Ь2в
H__inference_functional_1_layer_call_and_return_conditional_losses_632913
H__inference_functional_1_layer_call_and_return_conditional_losses_632809
H__inference_functional_1_layer_call_and_return_conditional_losses_632794
H__inference_functional_1_layer_call_and_return_conditional_losses_632939└
и▓│
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
▀2▄
!__inference__wrapped_model_632450Х
І▓Є
FullArgSpec
argsџ 
varargsjargs
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *&б#
!і
input_1         	
ѓ2 
-__inference_functional_1_layer_call_fn_632954
-__inference_functional_1_layer_call_fn_632840
-__inference_functional_1_layer_call_fn_632969
-__inference_functional_1_layer_call_fn_632870└
и▓│
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
ч2Э
L__inference_classifier_graph_layer_call_and_return_conditional_losses_633077
L__inference_classifier_graph_layer_call_and_return_conditional_losses_633021
L__inference_classifier_graph_layer_call_and_return_conditional_losses_633103
L__inference_classifier_graph_layer_call_and_return_conditional_losses_632995й
┤▓░
FullArgSpec/
args'џ$
jself
jx
	jpredict

jtraining
varargs
 
varkw
 
defaultsџ
p 
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Ј2ї
1__inference_classifier_graph_layer_call_fn_633051
1__inference_classifier_graph_layer_call_fn_633118
1__inference_classifier_graph_layer_call_fn_633133
1__inference_classifier_graph_layer_call_fn_633036й
┤▓░
FullArgSpec/
args'џ$
jself
jx
	jpredict

jtraining
varargs
 
varkw
 
defaultsџ
p 
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
3B1
$__inference_signature_wrapper_632887input_1
Т2с
F__inference_sequential_layer_call_and_return_conditional_losses_633185
F__inference_sequential_layer_call_and_return_conditional_losses_633267
F__inference_sequential_layer_call_and_return_conditional_losses_633241
F__inference_sequential_layer_call_and_return_conditional_losses_633159└
и▓│
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
Щ2э
+__inference_sequential_layer_call_fn_633297
+__inference_sequential_layer_call_fn_633282
+__inference_sequential_layer_call_fn_633215
+__inference_sequential_layer_call_fn_633200└
и▓│
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
Ѓ2ђ
C__inference_project_layer_call_and_return_conditional_losses_632463И
ћ▓љ
FullArgSpec
argsџ
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *б
і         	
У2т
(__inference_project_layer_call_fn_632471И
ћ▓љ
FullArgSpec
argsџ
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *б
і         	
ь2Ж
C__inference_layer-1_layer_call_and_return_conditional_losses_633308б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
м2¤
(__inference_layer-1_layer_call_fn_633317б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
В2ж
B__inference_output_layer_call_and_return_conditional_losses_633328б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Л2╬
'__inference_output_layer_call_fn_633337б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 Б
!__inference__wrapped_model_632450~0б-
&б#
!і
input_1         	
ф "Cф@
>
classifier_graph*і'
classifier_graph         И
L__inference_classifier_graph_layer_call_and_return_conditional_losses_632995h8б5
.б+
!і
input_1         	
p 
p
ф "%б"
і
0         
џ И
L__inference_classifier_graph_layer_call_and_return_conditional_losses_633021h8б5
.б+
!і
input_1         	
p 
p 
ф "%б"
і
0         
џ ▓
L__inference_classifier_graph_layer_call_and_return_conditional_losses_633077b2б/
(б%
і
x         	
p 
p
ф "%б"
і
0         
џ ▓
L__inference_classifier_graph_layer_call_and_return_conditional_losses_633103b2б/
(б%
і
x         	
p 
p 
ф "%б"
і
0         
џ љ
1__inference_classifier_graph_layer_call_fn_633036[8б5
.б+
!і
input_1         	
p 
p
ф "і         љ
1__inference_classifier_graph_layer_call_fn_633051[8б5
.б+
!і
input_1         	
p 
p 
ф "і         і
1__inference_classifier_graph_layer_call_fn_633118U2б/
(б%
і
x         	
p 
p
ф "і         і
1__inference_classifier_graph_layer_call_fn_633133U2б/
(б%
і
x         	
p 
p 
ф "і         ┤
H__inference_functional_1_layer_call_and_return_conditional_losses_632794h8б5
.б+
!і
input_1         	
p

 
ф "%б"
і
0         
џ ┤
H__inference_functional_1_layer_call_and_return_conditional_losses_632809h8б5
.б+
!і
input_1         	
p 

 
ф "%б"
і
0         
џ │
H__inference_functional_1_layer_call_and_return_conditional_losses_632913g7б4
-б*
 і
inputs         	
p

 
ф "%б"
і
0         
џ │
H__inference_functional_1_layer_call_and_return_conditional_losses_632939g7б4
-б*
 і
inputs         	
p 

 
ф "%б"
і
0         
џ ї
-__inference_functional_1_layer_call_fn_632840[8б5
.б+
!і
input_1         	
p

 
ф "і         ї
-__inference_functional_1_layer_call_fn_632870[8б5
.б+
!і
input_1         	
p 

 
ф "і         І
-__inference_functional_1_layer_call_fn_632954Z7б4
-б*
 і
inputs         	
p

 
ф "і         І
-__inference_functional_1_layer_call_fn_632969Z7б4
-б*
 і
inputs         	
p 

 
ф "і         Б
C__inference_layer-1_layer_call_and_return_conditional_losses_633308\/б,
%б"
 і
inputs         	
ф "%б"
і
0         2
џ {
(__inference_layer-1_layer_call_fn_633317O/б,
%б"
 і
inputs         	
ф "і         2б
B__inference_output_layer_call_and_return_conditional_losses_633328\/б,
%б"
 і
inputs         2
ф "%б"
і
0         
џ z
'__inference_output_layer_call_fn_633337O/б,
%б"
 і
inputs         2
ф "і         Ю
C__inference_project_layer_call_and_return_conditional_losses_632463V*б'
 б
і
x         	
ф "%б"
і
0         	
џ u
(__inference_project_layer_call_fn_632471I*б'
 б
і
x         	
ф "і         	▒
F__inference_sequential_layer_call_and_return_conditional_losses_633159g7б4
-б*
 і
inputs         	
p

 
ф "%б"
і
0         
џ ▒
F__inference_sequential_layer_call_and_return_conditional_losses_633185g7б4
-б*
 і
inputs         	
p 

 
ф "%б"
і
0         
џ И
F__inference_sequential_layer_call_and_return_conditional_losses_633241n>б;
4б1
'і$
project_input         	
p

 
ф "%б"
і
0         
џ И
F__inference_sequential_layer_call_and_return_conditional_losses_633267n>б;
4б1
'і$
project_input         	
p 

 
ф "%б"
і
0         
џ Ѕ
+__inference_sequential_layer_call_fn_633200Z7б4
-б*
 і
inputs         	
p

 
ф "і         Ѕ
+__inference_sequential_layer_call_fn_633215Z7б4
-б*
 і
inputs         	
p 

 
ф "і         љ
+__inference_sequential_layer_call_fn_633282a>б;
4б1
'і$
project_input         	
p

 
ф "і         љ
+__inference_sequential_layer_call_fn_633297a>б;
4б1
'і$
project_input         	
p 

 
ф "і         ▓
$__inference_signature_wrapper_632887Ѕ;б8
б 
1ф.
,
input_1!і
input_1         	"Cф@
>
classifier_graph*і'
classifier_graph         