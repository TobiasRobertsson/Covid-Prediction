??
??
D
AddV2
x"T
y"T
z"T"
Ttype:
2	??
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
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
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
?
Mul
x"T
y"T
z"T"
Ttype:
2	?
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
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
[
Split
	split_dim

value"T
output"T*	num_split"
	num_splitint(0"	
Ttype
?
SplitV

value"T
size_splits"Tlen
	split_dim
output"T*	num_split"
	num_splitint(0"	
Ttype"
Tlentype0	:
2	
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
executor_typestring ??
@
StaticRegexFullMatch	
input

output
"
patternstring
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	
-
Tanh
x"T
y"T"
Ttype:

2
?
TensorListFromTensor
tensor"element_dtype
element_shape"
shape_type*
output_handle??element_dtype"
element_dtypetype"

shape_typetype:
2	
?
TensorListReserve
element_shape"
shape_type
num_elements#
handle??element_dtype"
element_dtypetype"

shape_typetype:
2	
?
TensorListStack
input_handle
element_shape
tensor"element_dtype"
element_dtypetype" 
num_elementsint?????????
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
 ?
?
While

input2T
output2T"
T
list(type)("
condfunc"
bodyfunc" 
output_shapeslist(shape)
 "
parallel_iterationsint
?"serve*2.7.02v2.7.0-rc1-69-gc256c071bb28??
x
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namedense_1/kernel
q
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes

:*
dtype0
p
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_1/bias
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
?
gru_1/gru_cell_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:<*(
shared_namegru_1/gru_cell_1/kernel
?
+gru_1/gru_cell_1/kernel/Read/ReadVariableOpReadVariableOpgru_1/gru_cell_1/kernel*
_output_shapes

:<*
dtype0
?
!gru_1/gru_cell_1/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:<*2
shared_name#!gru_1/gru_cell_1/recurrent_kernel
?
5gru_1/gru_cell_1/recurrent_kernel/Read/ReadVariableOpReadVariableOp!gru_1/gru_cell_1/recurrent_kernel*
_output_shapes

:<*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
?
Adam/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_nameAdam/dense_1/kernel/m

)Adam/dense_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/m*
_output_shapes

:*
dtype0
~
Adam/dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_1/bias/m
w
'Adam/dense_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/m*
_output_shapes
:*
dtype0
?
Adam/gru_1/gru_cell_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:<*/
shared_name Adam/gru_1/gru_cell_1/kernel/m
?
2Adam/gru_1/gru_cell_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/gru_1/gru_cell_1/kernel/m*
_output_shapes

:<*
dtype0
?
(Adam/gru_1/gru_cell_1/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:<*9
shared_name*(Adam/gru_1/gru_cell_1/recurrent_kernel/m
?
<Adam/gru_1/gru_cell_1/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp(Adam/gru_1/gru_cell_1/recurrent_kernel/m*
_output_shapes

:<*
dtype0
?
Adam/dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_nameAdam/dense_1/kernel/v

)Adam/dense_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/v*
_output_shapes

:*
dtype0
~
Adam/dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_1/bias/v
w
'Adam/dense_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/v*
_output_shapes
:*
dtype0
?
Adam/gru_1/gru_cell_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:<*/
shared_name Adam/gru_1/gru_cell_1/kernel/v
?
2Adam/gru_1/gru_cell_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/gru_1/gru_cell_1/kernel/v*
_output_shapes

:<*
dtype0
?
(Adam/gru_1/gru_cell_1/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:<*9
shared_name*(Adam/gru_1/gru_cell_1/recurrent_kernel/v
?
<Adam/gru_1/gru_cell_1/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp(Adam/gru_1/gru_cell_1/recurrent_kernel/v*
_output_shapes

:<*
dtype0

NoOpNoOp
?
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?
value?B? B?
?
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api
	
signatures
l

cell

state_spec
	variables
trainable_variables
regularization_losses
	keras_api
R
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
?
iter

beta_1

beta_2
	decay
learning_ratemDmEmF mGvHvIvJ vK

0
 1
2
3

0
 1
2
3
 
?
!non_trainable_variables

"layers
#metrics
$layer_regularization_losses
%layer_metrics
	variables
trainable_variables
regularization_losses
 
t

kernel
 recurrent_kernel
&	variables
'trainable_variables
(regularization_losses
)	keras_api
 

0
 1

0
 1
 
?

*states
+non_trainable_variables

,layers
-metrics
.layer_regularization_losses
/layer_metrics
	variables
trainable_variables
regularization_losses
 
 
 
?
0non_trainable_variables

1layers
2metrics
3layer_regularization_losses
4layer_metrics
	variables
trainable_variables
regularization_losses
ZX
VARIABLE_VALUEdense_1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
5non_trainable_variables

6layers
7metrics
8layer_regularization_losses
9layer_metrics
	variables
trainable_variables
regularization_losses
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEgru_1/gru_cell_1/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUE!gru_1/gru_cell_1/recurrent_kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE
 

0
1
2

:0
 
 

0
 1

0
 1
 
?
;non_trainable_variables

<layers
=metrics
>layer_regularization_losses
?layer_metrics
&	variables
'trainable_variables
(regularization_losses
 
 


0
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
4
	@total
	Acount
B	variables
C	keras_api
 
 
 
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

@0
A1

B	variables
}{
VARIABLE_VALUEAdam/dense_1/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_1/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/gru_1/gru_cell_1/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUE(Adam/gru_1/gru_cell_1/recurrent_kernel/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_1/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_1/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/gru_1/gru_cell_1/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUE(Adam/gru_1/gru_cell_1/recurrent_kernel/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_gru_1_inputPlaceholder*+
_output_shapes
:?????????*
dtype0* 
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_gru_1_inputgru_1/gru_cell_1/kernel!gru_1/gru_cell_1/recurrent_kerneldense_1/kerneldense_1/bias*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *,
f'R%
#__inference_signature_wrapper_10616
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp+gru_1/gru_cell_1/kernel/Read/ReadVariableOp5gru_1/gru_cell_1/recurrent_kernel/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp)Adam/dense_1/kernel/m/Read/ReadVariableOp'Adam/dense_1/bias/m/Read/ReadVariableOp2Adam/gru_1/gru_cell_1/kernel/m/Read/ReadVariableOp<Adam/gru_1/gru_cell_1/recurrent_kernel/m/Read/ReadVariableOp)Adam/dense_1/kernel/v/Read/ReadVariableOp'Adam/dense_1/bias/v/Read/ReadVariableOp2Adam/gru_1/gru_cell_1/kernel/v/Read/ReadVariableOp<Adam/gru_1/gru_cell_1/recurrent_kernel/v/Read/ReadVariableOpConst* 
Tin
2	*
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
GPU 2J 8? *'
f"R 
__inference__traced_save_11749
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_1/kerneldense_1/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rategru_1/gru_cell_1/kernel!gru_1/gru_cell_1/recurrent_kerneltotalcountAdam/dense_1/kernel/mAdam/dense_1/bias/mAdam/gru_1/gru_cell_1/kernel/m(Adam/gru_1/gru_cell_1/recurrent_kernel/mAdam/dense_1/kernel/vAdam/dense_1/bias/vAdam/gru_1/gru_cell_1/kernel/v(Adam/gru_1/gru_cell_1/recurrent_kernel/v*
Tin
2*
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
GPU 2J 8? **
f%R#
!__inference__traced_restore_11816??
?
?
#__inference_signature_wrapper_10616
gru_1_input
unknown:<
	unknown_0:<
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallgru_1_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *(
f#R!
__inference__wrapped_model_9829o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
+
_output_shapes
:?????????
%
_user_specified_namegru_1_input
?
?
G__inference_sequential_1_layer_call_and_return_conditional_losses_10541

inputs
gru_1_10529:<
gru_1_10531:<
dense_1_10535:
dense_1_10537:
identity??dense_1/StatefulPartitionedCall?!dropout_1/StatefulPartitionedCall?gru_1/StatefulPartitionedCall?
gru_1/StatefulPartitionedCallStatefulPartitionedCallinputsgru_1_10529gru_1_10531*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_gru_1_layer_call_and_return_conditional_losses_10504?
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall&gru_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_10351?
dense_1/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0dense_1_10535dense_1_10537*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_10303w
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp ^dense_1/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall^gru_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : : : 2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2>
gru_1/StatefulPartitionedCallgru_1/StatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?4
?
while_body_10423
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0C
1while_gru_cell_1_matmul_readvariableop_resource_0:<E
3while_gru_cell_1_matmul_1_readvariableop_resource_0:<
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorA
/while_gru_cell_1_matmul_readvariableop_resource:<C
1while_gru_cell_1_matmul_1_readvariableop_resource:<??&while/gru_cell_1/MatMul/ReadVariableOp?(while/gru_cell_1/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0?
&while/gru_cell_1/MatMul/ReadVariableOpReadVariableOp1while_gru_cell_1_matmul_readvariableop_resource_0*
_output_shapes

:<*
dtype0?
while/gru_cell_1/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/gru_cell_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<k
 while/gru_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
while/gru_cell_1/splitSplit)while/gru_cell_1/split/split_dim:output:0!while/gru_cell_1/MatMul:product:0*
T0*M
_output_shapes;
9:?????????:?????????:?????????*
	num_split?
(while/gru_cell_1/MatMul_1/ReadVariableOpReadVariableOp3while_gru_cell_1_matmul_1_readvariableop_resource_0*
_output_shapes

:<*
dtype0?
while/gru_cell_1/MatMul_1MatMulwhile_placeholder_20while/gru_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<k
while/gru_cell_1/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ????m
"while/gru_cell_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
while/gru_cell_1/split_1SplitV#while/gru_cell_1/MatMul_1:product:0while/gru_cell_1/Const:output:0+while/gru_cell_1/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:?????????:?????????:?????????*
	num_split?
while/gru_cell_1/addAddV2while/gru_cell_1/split:output:0!while/gru_cell_1/split_1:output:0*
T0*'
_output_shapes
:?????????o
while/gru_cell_1/SigmoidSigmoidwhile/gru_cell_1/add:z:0*
T0*'
_output_shapes
:??????????
while/gru_cell_1/add_1AddV2while/gru_cell_1/split:output:1!while/gru_cell_1/split_1:output:1*
T0*'
_output_shapes
:?????????s
while/gru_cell_1/Sigmoid_1Sigmoidwhile/gru_cell_1/add_1:z:0*
T0*'
_output_shapes
:??????????
while/gru_cell_1/mulMulwhile/gru_cell_1/Sigmoid_1:y:0!while/gru_cell_1/split_1:output:2*
T0*'
_output_shapes
:??????????
while/gru_cell_1/add_2AddV2while/gru_cell_1/split:output:2while/gru_cell_1/mul:z:0*
T0*'
_output_shapes
:?????????k
while/gru_cell_1/TanhTanhwhile/gru_cell_1/add_2:z:0*
T0*'
_output_shapes
:??????????
while/gru_cell_1/mul_1Mulwhile/gru_cell_1/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:?????????[
while/gru_cell_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
while/gru_cell_1/subSubwhile/gru_cell_1/sub/x:output:0while/gru_cell_1/Sigmoid:y:0*
T0*'
_output_shapes
:??????????
while/gru_cell_1/mul_2Mulwhile/gru_cell_1/sub:z:0while/gru_cell_1/Tanh:y:0*
T0*'
_output_shapes
:??????????
while/gru_cell_1/add_3AddV2while/gru_cell_1/mul_1:z:0while/gru_cell_1/mul_2:z:0*
T0*'
_output_shapes
:??????????
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_1/add_3:z:0*
_output_shapes
: *
element_dtype0:???M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: ?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: w
while/Identity_4Identitywhile/gru_cell_1/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:??????????

while/NoOpNoOp'^while/gru_cell_1/MatMul/ReadVariableOp)^while/gru_cell_1/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "h
1while_gru_cell_1_matmul_1_readvariableop_resource3while_gru_cell_1_matmul_1_readvariableop_resource_0"d
/while_gru_cell_1_matmul_readvariableop_resource1while_gru_cell_1_matmul_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#: : : : :?????????: : : : 2P
&while/gru_cell_1/MatMul/ReadVariableOp&while/gru_cell_1/MatMul/ReadVariableOp2T
(while/gru_cell_1/MatMul_1/ReadVariableOp(while/gru_cell_1/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
: 
?
?
while_body_9904
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0)
while_gru_cell_1_9926_0:<)
while_gru_cell_1_9928_0:<
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor'
while_gru_cell_1_9926:<'
while_gru_cell_1_9928:<??(while/gru_cell_1/StatefulPartitionedCall?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0?
(while/gru_cell_1/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_gru_cell_1_9926_0while_gru_cell_1_9928_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_gru_cell_1_layer_call_and_return_conditional_losses_9893?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder1while/gru_cell_1/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:???M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: ?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: ?
while/Identity_4Identity1while/gru_cell_1/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:?????????w

while/NoOpNoOp)^while/gru_cell_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "0
while_gru_cell_1_9926while_gru_cell_1_9926_0"0
while_gru_cell_1_9928while_gru_cell_1_9928_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#: : : : :?????????: : : : 2T
(while/gru_cell_1/StatefulPartitionedCall(while/gru_cell_1/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
: 
?
?
while_cond_11451
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_11451___redundant_placeholder03
/while_while_cond_11451___redundant_placeholder13
/while_while_cond_11451___redundant_placeholder2
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
): : : : :?????????: :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
:
?F
?
@__inference_gru_1_layer_call_and_return_conditional_losses_10504

inputs;
)gru_cell_1_matmul_readvariableop_resource:<=
+gru_cell_1_matmul_1_readvariableop_resource:<
identity?? gru_cell_1/MatMul/ReadVariableOp?"gru_cell_1/MatMul_1/ReadVariableOp?while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:?????????D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask?
 gru_cell_1/MatMul/ReadVariableOpReadVariableOp)gru_cell_1_matmul_readvariableop_resource*
_output_shapes

:<*
dtype0?
gru_cell_1/MatMulMatMulstrided_slice_2:output:0(gru_cell_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<e
gru_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
gru_cell_1/splitSplit#gru_cell_1/split/split_dim:output:0gru_cell_1/MatMul:product:0*
T0*M
_output_shapes;
9:?????????:?????????:?????????*
	num_split?
"gru_cell_1/MatMul_1/ReadVariableOpReadVariableOp+gru_cell_1_matmul_1_readvariableop_resource*
_output_shapes

:<*
dtype0?
gru_cell_1/MatMul_1MatMulzeros:output:0*gru_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<e
gru_cell_1/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ????g
gru_cell_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
gru_cell_1/split_1SplitVgru_cell_1/MatMul_1:product:0gru_cell_1/Const:output:0%gru_cell_1/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:?????????:?????????:?????????*
	num_split?
gru_cell_1/addAddV2gru_cell_1/split:output:0gru_cell_1/split_1:output:0*
T0*'
_output_shapes
:?????????c
gru_cell_1/SigmoidSigmoidgru_cell_1/add:z:0*
T0*'
_output_shapes
:??????????
gru_cell_1/add_1AddV2gru_cell_1/split:output:1gru_cell_1/split_1:output:1*
T0*'
_output_shapes
:?????????g
gru_cell_1/Sigmoid_1Sigmoidgru_cell_1/add_1:z:0*
T0*'
_output_shapes
:?????????~
gru_cell_1/mulMulgru_cell_1/Sigmoid_1:y:0gru_cell_1/split_1:output:2*
T0*'
_output_shapes
:?????????z
gru_cell_1/add_2AddV2gru_cell_1/split:output:2gru_cell_1/mul:z:0*
T0*'
_output_shapes
:?????????_
gru_cell_1/TanhTanhgru_cell_1/add_2:z:0*
T0*'
_output_shapes
:?????????q
gru_cell_1/mul_1Mulgru_cell_1/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:?????????U
gru_cell_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??z
gru_cell_1/subSubgru_cell_1/sub/x:output:0gru_cell_1/Sigmoid:y:0*
T0*'
_output_shapes
:?????????r
gru_cell_1/mul_2Mulgru_cell_1/sub:z:0gru_cell_1/Tanh:y:0*
T0*'
_output_shapes
:?????????w
gru_cell_1/add_3AddV2gru_cell_1/mul_1:z:0gru_cell_1/mul_2:z:0*
T0*'
_output_shapes
:?????????n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)gru_cell_1_matmul_readvariableop_resource+gru_cell_1_matmul_1_readvariableop_resource*
T
2	*
_lower_using_switch_merge(*
_num_original_outputs	*7
_output_shapes%
#: : : : :?????????: : : : *$
_read_only_resource_inputs
*
_stateful_parallelism( *
bodyR
while_body_10423*
condR
while_cond_10422*6
output_shapes%
#: : : : :?????????: : : : *
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp!^gru_cell_1/MatMul/ReadVariableOp#^gru_cell_1/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : 2D
 gru_cell_1/MatMul/ReadVariableOp gru_cell_1/MatMul/ReadVariableOp2H
"gru_cell_1/MatMul_1/ReadVariableOp"gru_cell_1/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
while_cond_9903
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_12
.while_while_cond_9903___redundant_placeholder02
.while_while_cond_9903___redundant_placeholder12
.while_while_cond_9903___redundant_placeholder2
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
): : : : :?????????: :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
:
?
?
%__inference_gru_1_layer_call_fn_10977

inputs
unknown:<
	unknown_0:<
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_gru_1_layer_call_and_return_conditional_losses_10504o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?4
?
while_body_11174
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0C
1while_gru_cell_1_matmul_readvariableop_resource_0:<E
3while_gru_cell_1_matmul_1_readvariableop_resource_0:<
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorA
/while_gru_cell_1_matmul_readvariableop_resource:<C
1while_gru_cell_1_matmul_1_readvariableop_resource:<??&while/gru_cell_1/MatMul/ReadVariableOp?(while/gru_cell_1/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0?
&while/gru_cell_1/MatMul/ReadVariableOpReadVariableOp1while_gru_cell_1_matmul_readvariableop_resource_0*
_output_shapes

:<*
dtype0?
while/gru_cell_1/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/gru_cell_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<k
 while/gru_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
while/gru_cell_1/splitSplit)while/gru_cell_1/split/split_dim:output:0!while/gru_cell_1/MatMul:product:0*
T0*M
_output_shapes;
9:?????????:?????????:?????????*
	num_split?
(while/gru_cell_1/MatMul_1/ReadVariableOpReadVariableOp3while_gru_cell_1_matmul_1_readvariableop_resource_0*
_output_shapes

:<*
dtype0?
while/gru_cell_1/MatMul_1MatMulwhile_placeholder_20while/gru_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<k
while/gru_cell_1/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ????m
"while/gru_cell_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
while/gru_cell_1/split_1SplitV#while/gru_cell_1/MatMul_1:product:0while/gru_cell_1/Const:output:0+while/gru_cell_1/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:?????????:?????????:?????????*
	num_split?
while/gru_cell_1/addAddV2while/gru_cell_1/split:output:0!while/gru_cell_1/split_1:output:0*
T0*'
_output_shapes
:?????????o
while/gru_cell_1/SigmoidSigmoidwhile/gru_cell_1/add:z:0*
T0*'
_output_shapes
:??????????
while/gru_cell_1/add_1AddV2while/gru_cell_1/split:output:1!while/gru_cell_1/split_1:output:1*
T0*'
_output_shapes
:?????????s
while/gru_cell_1/Sigmoid_1Sigmoidwhile/gru_cell_1/add_1:z:0*
T0*'
_output_shapes
:??????????
while/gru_cell_1/mulMulwhile/gru_cell_1/Sigmoid_1:y:0!while/gru_cell_1/split_1:output:2*
T0*'
_output_shapes
:??????????
while/gru_cell_1/add_2AddV2while/gru_cell_1/split:output:2while/gru_cell_1/mul:z:0*
T0*'
_output_shapes
:?????????k
while/gru_cell_1/TanhTanhwhile/gru_cell_1/add_2:z:0*
T0*'
_output_shapes
:??????????
while/gru_cell_1/mul_1Mulwhile/gru_cell_1/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:?????????[
while/gru_cell_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
while/gru_cell_1/subSubwhile/gru_cell_1/sub/x:output:0while/gru_cell_1/Sigmoid:y:0*
T0*'
_output_shapes
:??????????
while/gru_cell_1/mul_2Mulwhile/gru_cell_1/sub:z:0while/gru_cell_1/Tanh:y:0*
T0*'
_output_shapes
:??????????
while/gru_cell_1/add_3AddV2while/gru_cell_1/mul_1:z:0while/gru_cell_1/mul_2:z:0*
T0*'
_output_shapes
:??????????
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_1/add_3:z:0*
_output_shapes
: *
element_dtype0:???M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: ?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: w
while/Identity_4Identitywhile/gru_cell_1/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:??????????

while/NoOpNoOp'^while/gru_cell_1/MatMul/ReadVariableOp)^while/gru_cell_1/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "h
1while_gru_cell_1_matmul_1_readvariableop_resource3while_gru_cell_1_matmul_1_readvariableop_resource_0"d
/while_gru_cell_1_matmul_readvariableop_resource1while_gru_cell_1_matmul_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#: : : : :?????????: : : : 2P
&while/gru_cell_1/MatMul/ReadVariableOp&while/gru_cell_1/MatMul/ReadVariableOp2T
(while/gru_cell_1/MatMul_1/ReadVariableOp(while/gru_cell_1/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
: 
?
b
)__inference_dropout_1_layer_call_fn_11543

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_10351o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
D__inference_gru_cell_1_layer_call_and_return_conditional_losses_9893

inputs

states0
matmul_readvariableop_resource:<2
 matmul_1_readvariableop_resource:<
identity

identity_1??MatMul/ReadVariableOp?MatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:<*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<Z
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
splitSplitsplit/split_dim:output:0MatMul:product:0*
T0*M
_output_shapes;
9:?????????:?????????:?????????*
	num_splitx
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:<*
dtype0m
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<Z
ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ????\
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
split_1SplitVMatMul_1:product:0Const:output:0split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:?????????:?????????:?????????*
	num_split`
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:?????????M
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:?????????b
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:?????????Q
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:?????????]
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:?????????Y
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:?????????I
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:?????????S
mul_1MulSigmoid:y:0states*
T0*'
_output_shapes
:?????????J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:?????????Q
mul_2Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:?????????V
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:?????????X
IdentityIdentity	add_3:z:0^NoOp*
T0*'
_output_shapes
:?????????Z

Identity_1Identity	add_3:z:0^NoOp*
T0*'
_output_shapes
:?????????x
NoOpNoOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*=
_input_shapes,
*:?????????:?????????: : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_namestates
?G
?
@__inference_gru_1_layer_call_and_return_conditional_losses_11255
inputs_0;
)gru_cell_1_matmul_readvariableop_resource:<=
+gru_cell_1_matmul_1_readvariableop_resource:<
identity?? gru_cell_1/MatMul/ReadVariableOp?"gru_cell_1/MatMul_1/ReadVariableOp?while=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask?
 gru_cell_1/MatMul/ReadVariableOpReadVariableOp)gru_cell_1_matmul_readvariableop_resource*
_output_shapes

:<*
dtype0?
gru_cell_1/MatMulMatMulstrided_slice_2:output:0(gru_cell_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<e
gru_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
gru_cell_1/splitSplit#gru_cell_1/split/split_dim:output:0gru_cell_1/MatMul:product:0*
T0*M
_output_shapes;
9:?????????:?????????:?????????*
	num_split?
"gru_cell_1/MatMul_1/ReadVariableOpReadVariableOp+gru_cell_1_matmul_1_readvariableop_resource*
_output_shapes

:<*
dtype0?
gru_cell_1/MatMul_1MatMulzeros:output:0*gru_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<e
gru_cell_1/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ????g
gru_cell_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
gru_cell_1/split_1SplitVgru_cell_1/MatMul_1:product:0gru_cell_1/Const:output:0%gru_cell_1/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:?????????:?????????:?????????*
	num_split?
gru_cell_1/addAddV2gru_cell_1/split:output:0gru_cell_1/split_1:output:0*
T0*'
_output_shapes
:?????????c
gru_cell_1/SigmoidSigmoidgru_cell_1/add:z:0*
T0*'
_output_shapes
:??????????
gru_cell_1/add_1AddV2gru_cell_1/split:output:1gru_cell_1/split_1:output:1*
T0*'
_output_shapes
:?????????g
gru_cell_1/Sigmoid_1Sigmoidgru_cell_1/add_1:z:0*
T0*'
_output_shapes
:?????????~
gru_cell_1/mulMulgru_cell_1/Sigmoid_1:y:0gru_cell_1/split_1:output:2*
T0*'
_output_shapes
:?????????z
gru_cell_1/add_2AddV2gru_cell_1/split:output:2gru_cell_1/mul:z:0*
T0*'
_output_shapes
:?????????_
gru_cell_1/TanhTanhgru_cell_1/add_2:z:0*
T0*'
_output_shapes
:?????????q
gru_cell_1/mul_1Mulgru_cell_1/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:?????????U
gru_cell_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??z
gru_cell_1/subSubgru_cell_1/sub/x:output:0gru_cell_1/Sigmoid:y:0*
T0*'
_output_shapes
:?????????r
gru_cell_1/mul_2Mulgru_cell_1/sub:z:0gru_cell_1/Tanh:y:0*
T0*'
_output_shapes
:?????????w
gru_cell_1/add_3AddV2gru_cell_1/mul_1:z:0gru_cell_1/mul_2:z:0*
T0*'
_output_shapes
:?????????n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)gru_cell_1_matmul_readvariableop_resource+gru_cell_1_matmul_1_readvariableop_resource*
T
2	*
_lower_using_switch_merge(*
_num_original_outputs	*7
_output_shapes%
#: : : : :?????????: : : : *$
_read_only_resource_inputs
*
_stateful_parallelism( *
bodyR
while_body_11174*
condR
while_cond_11173*6
output_shapes%
#: : : : :?????????: : : : *
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp!^gru_cell_1/MatMul/ReadVariableOp#^gru_cell_1/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????????????: : 2D
 gru_cell_1/MatMul/ReadVariableOp gru_cell_1/MatMul/ReadVariableOp2H
"gru_cell_1/MatMul_1/ReadVariableOp"gru_cell_1/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :??????????????????
"
_user_specified_name
inputs/0
?N
?
!__inference__traced_restore_11816
file_prefix1
assignvariableop_dense_1_kernel:-
assignvariableop_1_dense_1_bias:&
assignvariableop_2_adam_iter:	 (
assignvariableop_3_adam_beta_1: (
assignvariableop_4_adam_beta_2: '
assignvariableop_5_adam_decay: /
%assignvariableop_6_adam_learning_rate: <
*assignvariableop_7_gru_1_gru_cell_1_kernel:<F
4assignvariableop_8_gru_1_gru_cell_1_recurrent_kernel:<"
assignvariableop_9_total: #
assignvariableop_10_count: ;
)assignvariableop_11_adam_dense_1_kernel_m:5
'assignvariableop_12_adam_dense_1_bias_m:D
2assignvariableop_13_adam_gru_1_gru_cell_1_kernel_m:<N
<assignvariableop_14_adam_gru_1_gru_cell_1_recurrent_kernel_m:<;
)assignvariableop_15_adam_dense_1_kernel_v:5
'assignvariableop_16_adam_dense_1_bias_v:D
2assignvariableop_17_adam_gru_1_gru_cell_1_kernel_v:<N
<assignvariableop_18_adam_gru_1_gru_cell_1_recurrent_kernel_v:<
identity_20??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_2?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?	
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?	
value?	B?	B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*;
value2B0B B B B B B B B B B B B B B B B B B B B ?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*d
_output_shapesR
P::::::::::::::::::::*"
dtypes
2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOpAssignVariableOpassignvariableop_dense_1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0	*
_output_shapes
:?
AssignVariableOp_2AssignVariableOpassignvariableop_2_adam_iterIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_3AssignVariableOpassignvariableop_3_adam_beta_1Identity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_beta_2Identity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_decayIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_6AssignVariableOp%assignvariableop_6_adam_learning_rateIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_7AssignVariableOp*assignvariableop_7_gru_1_gru_cell_1_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_8AssignVariableOp4assignvariableop_8_gru_1_gru_cell_1_recurrent_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_9AssignVariableOpassignvariableop_9_totalIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_10AssignVariableOpassignvariableop_10_countIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_11AssignVariableOp)assignvariableop_11_adam_dense_1_kernel_mIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_12AssignVariableOp'assignvariableop_12_adam_dense_1_bias_mIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_13AssignVariableOp2assignvariableop_13_adam_gru_1_gru_cell_1_kernel_mIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_14AssignVariableOp<assignvariableop_14_adam_gru_1_gru_cell_1_recurrent_kernel_mIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_15AssignVariableOp)assignvariableop_15_adam_dense_1_kernel_vIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_16AssignVariableOp'assignvariableop_16_adam_dense_1_bias_vIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_17AssignVariableOp2assignvariableop_17_adam_gru_1_gru_cell_1_kernel_vIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_18AssignVariableOp<assignvariableop_18_adam_gru_1_gru_cell_1_recurrent_kernel_vIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ?
Identity_19Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_20IdentityIdentity_19:output:0^NoOp_1*
T0*
_output_shapes
: ?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_20Identity_20:output:0*;
_input_shapes*
(: : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?4
?
while_body_11035
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0C
1while_gru_cell_1_matmul_readvariableop_resource_0:<E
3while_gru_cell_1_matmul_1_readvariableop_resource_0:<
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorA
/while_gru_cell_1_matmul_readvariableop_resource:<C
1while_gru_cell_1_matmul_1_readvariableop_resource:<??&while/gru_cell_1/MatMul/ReadVariableOp?(while/gru_cell_1/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0?
&while/gru_cell_1/MatMul/ReadVariableOpReadVariableOp1while_gru_cell_1_matmul_readvariableop_resource_0*
_output_shapes

:<*
dtype0?
while/gru_cell_1/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/gru_cell_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<k
 while/gru_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
while/gru_cell_1/splitSplit)while/gru_cell_1/split/split_dim:output:0!while/gru_cell_1/MatMul:product:0*
T0*M
_output_shapes;
9:?????????:?????????:?????????*
	num_split?
(while/gru_cell_1/MatMul_1/ReadVariableOpReadVariableOp3while_gru_cell_1_matmul_1_readvariableop_resource_0*
_output_shapes

:<*
dtype0?
while/gru_cell_1/MatMul_1MatMulwhile_placeholder_20while/gru_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<k
while/gru_cell_1/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ????m
"while/gru_cell_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
while/gru_cell_1/split_1SplitV#while/gru_cell_1/MatMul_1:product:0while/gru_cell_1/Const:output:0+while/gru_cell_1/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:?????????:?????????:?????????*
	num_split?
while/gru_cell_1/addAddV2while/gru_cell_1/split:output:0!while/gru_cell_1/split_1:output:0*
T0*'
_output_shapes
:?????????o
while/gru_cell_1/SigmoidSigmoidwhile/gru_cell_1/add:z:0*
T0*'
_output_shapes
:??????????
while/gru_cell_1/add_1AddV2while/gru_cell_1/split:output:1!while/gru_cell_1/split_1:output:1*
T0*'
_output_shapes
:?????????s
while/gru_cell_1/Sigmoid_1Sigmoidwhile/gru_cell_1/add_1:z:0*
T0*'
_output_shapes
:??????????
while/gru_cell_1/mulMulwhile/gru_cell_1/Sigmoid_1:y:0!while/gru_cell_1/split_1:output:2*
T0*'
_output_shapes
:??????????
while/gru_cell_1/add_2AddV2while/gru_cell_1/split:output:2while/gru_cell_1/mul:z:0*
T0*'
_output_shapes
:?????????k
while/gru_cell_1/TanhTanhwhile/gru_cell_1/add_2:z:0*
T0*'
_output_shapes
:??????????
while/gru_cell_1/mul_1Mulwhile/gru_cell_1/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:?????????[
while/gru_cell_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
while/gru_cell_1/subSubwhile/gru_cell_1/sub/x:output:0while/gru_cell_1/Sigmoid:y:0*
T0*'
_output_shapes
:??????????
while/gru_cell_1/mul_2Mulwhile/gru_cell_1/sub:z:0while/gru_cell_1/Tanh:y:0*
T0*'
_output_shapes
:??????????
while/gru_cell_1/add_3AddV2while/gru_cell_1/mul_1:z:0while/gru_cell_1/mul_2:z:0*
T0*'
_output_shapes
:??????????
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_1/add_3:z:0*
_output_shapes
: *
element_dtype0:???M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: ?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: w
while/Identity_4Identitywhile/gru_cell_1/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:??????????

while/NoOpNoOp'^while/gru_cell_1/MatMul/ReadVariableOp)^while/gru_cell_1/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "h
1while_gru_cell_1_matmul_1_readvariableop_resource3while_gru_cell_1_matmul_1_readvariableop_resource_0"d
/while_gru_cell_1_matmul_readvariableop_resource1while_gru_cell_1_matmul_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#: : : : :?????????: : : : 2P
&while/gru_cell_1/MatMul/ReadVariableOp&while/gru_cell_1/MatMul/ReadVariableOp2T
(while/gru_cell_1/MatMul_1/ReadVariableOp(while/gru_cell_1/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
: 
?
?
'__inference_dense_1_layer_call_fn_11569

inputs
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_10303o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
while_cond_11034
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_11034___redundant_placeholder03
/while_while_cond_11034___redundant_placeholder13
/while_while_cond_11034___redundant_placeholder2
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
): : : : :?????????: :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
:
?
?
while_body_10068
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*
while_gru_cell_1_10090_0:<*
while_gru_cell_1_10092_0:<
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor(
while_gru_cell_1_10090:<(
while_gru_cell_1_10092:<??(while/gru_cell_1/StatefulPartitionedCall?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0?
(while/gru_cell_1/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_gru_cell_1_10090_0while_gru_cell_1_10092_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_gru_cell_1_layer_call_and_return_conditional_losses_10020?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder1while/gru_cell_1/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:???M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: ?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: ?
while/Identity_4Identity1while/gru_cell_1/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:?????????w

while/NoOpNoOp)^while/gru_cell_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "2
while_gru_cell_1_10090while_gru_cell_1_10090_0"2
while_gru_cell_1_10092while_gru_cell_1_10092_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#: : : : :?????????: : : : 2T
(while/gru_cell_1/StatefulPartitionedCall(while/gru_cell_1/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
: 
?
?
while_cond_10422
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_10422___redundant_placeholder03
/while_while_cond_10422___redundant_placeholder13
/while_while_cond_10422___redundant_placeholder2
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
): : : : :?????????: :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
:
?U
?
G__inference_sequential_1_layer_call_and_return_conditional_losses_10788

inputsA
/gru_1_gru_cell_1_matmul_readvariableop_resource:<C
1gru_1_gru_cell_1_matmul_1_readvariableop_resource:<8
&dense_1_matmul_readvariableop_resource:5
'dense_1_biasadd_readvariableop_resource:
identity??dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?&gru_1/gru_cell_1/MatMul/ReadVariableOp?(gru_1/gru_cell_1/MatMul_1/ReadVariableOp?gru_1/whileA
gru_1/ShapeShapeinputs*
T0*
_output_shapes
:c
gru_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: e
gru_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:e
gru_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
gru_1/strided_sliceStridedSlicegru_1/Shape:output:0"gru_1/strided_slice/stack:output:0$gru_1/strided_slice/stack_1:output:0$gru_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskV
gru_1/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :?
gru_1/zeros/packedPackgru_1/strided_slice:output:0gru_1/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:V
gru_1/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ~
gru_1/zerosFillgru_1/zeros/packed:output:0gru_1/zeros/Const:output:0*
T0*'
_output_shapes
:?????????i
gru_1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          y
gru_1/transpose	Transposeinputsgru_1/transpose/perm:output:0*
T0*+
_output_shapes
:?????????P
gru_1/Shape_1Shapegru_1/transpose:y:0*
T0*
_output_shapes
:e
gru_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: g
gru_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
gru_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
gru_1/strided_slice_1StridedSlicegru_1/Shape_1:output:0$gru_1/strided_slice_1/stack:output:0&gru_1/strided_slice_1/stack_1:output:0&gru_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskl
!gru_1/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
gru_1/TensorArrayV2TensorListReserve*gru_1/TensorArrayV2/element_shape:output:0gru_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
;gru_1/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
-gru_1/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorgru_1/transpose:y:0Dgru_1/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???e
gru_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: g
gru_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
gru_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
gru_1/strided_slice_2StridedSlicegru_1/transpose:y:0$gru_1/strided_slice_2/stack:output:0&gru_1/strided_slice_2/stack_1:output:0&gru_1/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask?
&gru_1/gru_cell_1/MatMul/ReadVariableOpReadVariableOp/gru_1_gru_cell_1_matmul_readvariableop_resource*
_output_shapes

:<*
dtype0?
gru_1/gru_cell_1/MatMulMatMulgru_1/strided_slice_2:output:0.gru_1/gru_cell_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<k
 gru_1/gru_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
gru_1/gru_cell_1/splitSplit)gru_1/gru_cell_1/split/split_dim:output:0!gru_1/gru_cell_1/MatMul:product:0*
T0*M
_output_shapes;
9:?????????:?????????:?????????*
	num_split?
(gru_1/gru_cell_1/MatMul_1/ReadVariableOpReadVariableOp1gru_1_gru_cell_1_matmul_1_readvariableop_resource*
_output_shapes

:<*
dtype0?
gru_1/gru_cell_1/MatMul_1MatMulgru_1/zeros:output:00gru_1/gru_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<k
gru_1/gru_cell_1/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ????m
"gru_1/gru_cell_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
gru_1/gru_cell_1/split_1SplitV#gru_1/gru_cell_1/MatMul_1:product:0gru_1/gru_cell_1/Const:output:0+gru_1/gru_cell_1/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:?????????:?????????:?????????*
	num_split?
gru_1/gru_cell_1/addAddV2gru_1/gru_cell_1/split:output:0!gru_1/gru_cell_1/split_1:output:0*
T0*'
_output_shapes
:?????????o
gru_1/gru_cell_1/SigmoidSigmoidgru_1/gru_cell_1/add:z:0*
T0*'
_output_shapes
:??????????
gru_1/gru_cell_1/add_1AddV2gru_1/gru_cell_1/split:output:1!gru_1/gru_cell_1/split_1:output:1*
T0*'
_output_shapes
:?????????s
gru_1/gru_cell_1/Sigmoid_1Sigmoidgru_1/gru_cell_1/add_1:z:0*
T0*'
_output_shapes
:??????????
gru_1/gru_cell_1/mulMulgru_1/gru_cell_1/Sigmoid_1:y:0!gru_1/gru_cell_1/split_1:output:2*
T0*'
_output_shapes
:??????????
gru_1/gru_cell_1/add_2AddV2gru_1/gru_cell_1/split:output:2gru_1/gru_cell_1/mul:z:0*
T0*'
_output_shapes
:?????????k
gru_1/gru_cell_1/TanhTanhgru_1/gru_cell_1/add_2:z:0*
T0*'
_output_shapes
:??????????
gru_1/gru_cell_1/mul_1Mulgru_1/gru_cell_1/Sigmoid:y:0gru_1/zeros:output:0*
T0*'
_output_shapes
:?????????[
gru_1/gru_cell_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
gru_1/gru_cell_1/subSubgru_1/gru_cell_1/sub/x:output:0gru_1/gru_cell_1/Sigmoid:y:0*
T0*'
_output_shapes
:??????????
gru_1/gru_cell_1/mul_2Mulgru_1/gru_cell_1/sub:z:0gru_1/gru_cell_1/Tanh:y:0*
T0*'
_output_shapes
:??????????
gru_1/gru_cell_1/add_3AddV2gru_1/gru_cell_1/mul_1:z:0gru_1/gru_cell_1/mul_2:z:0*
T0*'
_output_shapes
:?????????t
#gru_1/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
gru_1/TensorArrayV2_1TensorListReserve,gru_1/TensorArrayV2_1/element_shape:output:0gru_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???L

gru_1/timeConst*
_output_shapes
: *
dtype0*
value	B : i
gru_1/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????Z
gru_1/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
gru_1/whileWhile!gru_1/while/loop_counter:output:0'gru_1/while/maximum_iterations:output:0gru_1/time:output:0gru_1/TensorArrayV2_1:handle:0gru_1/zeros:output:0gru_1/strided_slice_1:output:0=gru_1/TensorArrayUnstack/TensorListFromTensor:output_handle:0/gru_1_gru_cell_1_matmul_readvariableop_resource1gru_1_gru_cell_1_matmul_1_readvariableop_resource*
T
2	*
_lower_using_switch_merge(*
_num_original_outputs	*7
_output_shapes%
#: : : : :?????????: : : : *$
_read_only_resource_inputs
*
_stateful_parallelism( *"
bodyR
gru_1_while_body_10700*"
condR
gru_1_while_cond_10699*6
output_shapes%
#: : : : :?????????: : : : *
parallel_iterations ?
6gru_1/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
(gru_1/TensorArrayV2Stack/TensorListStackTensorListStackgru_1/while:output:3?gru_1/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????*
element_dtype0n
gru_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????g
gru_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: g
gru_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
gru_1/strided_slice_3StridedSlice1gru_1/TensorArrayV2Stack/TensorListStack:tensor:0$gru_1/strided_slice_3/stack:output:0&gru_1/strided_slice_3/stack_1:output:0&gru_1/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_maskk
gru_1/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
gru_1/transpose_1	Transpose1gru_1/TensorArrayV2Stack/TensorListStack:tensor:0gru_1/transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????a
gru_1/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    p
dropout_1/IdentityIdentitygru_1/strided_slice_3:output:0*
T0*'
_output_shapes
:??????????
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
dense_1/MatMulMatMuldropout_1/Identity:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????g
IdentityIdentitydense_1/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp'^gru_1/gru_cell_1/MatMul/ReadVariableOp)^gru_1/gru_cell_1/MatMul_1/ReadVariableOp^gru_1/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : : : 2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2P
&gru_1/gru_cell_1/MatMul/ReadVariableOp&gru_1/gru_cell_1/MatMul/ReadVariableOp2T
(gru_1/gru_cell_1/MatMul_1/ReadVariableOp(gru_1/gru_cell_1/MatMul_1/ReadVariableOp2
gru_1/whilegru_1/while:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?F
?
@__inference_gru_1_layer_call_and_return_conditional_losses_11394

inputs;
)gru_cell_1_matmul_readvariableop_resource:<=
+gru_cell_1_matmul_1_readvariableop_resource:<
identity?? gru_cell_1/MatMul/ReadVariableOp?"gru_cell_1/MatMul_1/ReadVariableOp?while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:?????????D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask?
 gru_cell_1/MatMul/ReadVariableOpReadVariableOp)gru_cell_1_matmul_readvariableop_resource*
_output_shapes

:<*
dtype0?
gru_cell_1/MatMulMatMulstrided_slice_2:output:0(gru_cell_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<e
gru_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
gru_cell_1/splitSplit#gru_cell_1/split/split_dim:output:0gru_cell_1/MatMul:product:0*
T0*M
_output_shapes;
9:?????????:?????????:?????????*
	num_split?
"gru_cell_1/MatMul_1/ReadVariableOpReadVariableOp+gru_cell_1_matmul_1_readvariableop_resource*
_output_shapes

:<*
dtype0?
gru_cell_1/MatMul_1MatMulzeros:output:0*gru_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<e
gru_cell_1/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ????g
gru_cell_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
gru_cell_1/split_1SplitVgru_cell_1/MatMul_1:product:0gru_cell_1/Const:output:0%gru_cell_1/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:?????????:?????????:?????????*
	num_split?
gru_cell_1/addAddV2gru_cell_1/split:output:0gru_cell_1/split_1:output:0*
T0*'
_output_shapes
:?????????c
gru_cell_1/SigmoidSigmoidgru_cell_1/add:z:0*
T0*'
_output_shapes
:??????????
gru_cell_1/add_1AddV2gru_cell_1/split:output:1gru_cell_1/split_1:output:1*
T0*'
_output_shapes
:?????????g
gru_cell_1/Sigmoid_1Sigmoidgru_cell_1/add_1:z:0*
T0*'
_output_shapes
:?????????~
gru_cell_1/mulMulgru_cell_1/Sigmoid_1:y:0gru_cell_1/split_1:output:2*
T0*'
_output_shapes
:?????????z
gru_cell_1/add_2AddV2gru_cell_1/split:output:2gru_cell_1/mul:z:0*
T0*'
_output_shapes
:?????????_
gru_cell_1/TanhTanhgru_cell_1/add_2:z:0*
T0*'
_output_shapes
:?????????q
gru_cell_1/mul_1Mulgru_cell_1/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:?????????U
gru_cell_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??z
gru_cell_1/subSubgru_cell_1/sub/x:output:0gru_cell_1/Sigmoid:y:0*
T0*'
_output_shapes
:?????????r
gru_cell_1/mul_2Mulgru_cell_1/sub:z:0gru_cell_1/Tanh:y:0*
T0*'
_output_shapes
:?????????w
gru_cell_1/add_3AddV2gru_cell_1/mul_1:z:0gru_cell_1/mul_2:z:0*
T0*'
_output_shapes
:?????????n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)gru_cell_1_matmul_readvariableop_resource+gru_cell_1_matmul_1_readvariableop_resource*
T
2	*
_lower_using_switch_merge(*
_num_original_outputs	*7
_output_shapes%
#: : : : :?????????: : : : *$
_read_only_resource_inputs
*
_stateful_parallelism( *
bodyR
while_body_11313*
condR
while_cond_11312*6
output_shapes%
#: : : : :?????????: : : : *
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp!^gru_cell_1/MatMul/ReadVariableOp#^gru_cell_1/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : 2D
 gru_cell_1/MatMul/ReadVariableOp gru_cell_1/MatMul/ReadVariableOp2H
"gru_cell_1/MatMul_1/ReadVariableOp"gru_cell_1/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
E__inference_gru_cell_1_layer_call_and_return_conditional_losses_10020

inputs

states0
matmul_readvariableop_resource:<2
 matmul_1_readvariableop_resource:<
identity

identity_1??MatMul/ReadVariableOp?MatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:<*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<Z
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
splitSplitsplit/split_dim:output:0MatMul:product:0*
T0*M
_output_shapes;
9:?????????:?????????:?????????*
	num_splitx
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:<*
dtype0m
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<Z
ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ????\
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
split_1SplitVMatMul_1:product:0Const:output:0split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:?????????:?????????:?????????*
	num_split`
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:?????????M
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:?????????b
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:?????????Q
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:?????????]
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:?????????Y
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:?????????I
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:?????????S
mul_1MulSigmoid:y:0states*
T0*'
_output_shapes
:?????????J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:?????????Q
mul_2Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:?????????V
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:?????????X
IdentityIdentity	add_3:z:0^NoOp*
T0*'
_output_shapes
:?????????Z

Identity_1Identity	add_3:z:0^NoOp*
T0*'
_output_shapes
:?????????x
NoOpNoOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*=
_input_shapes,
*:?????????:?????????: : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_namestates
?f
?
__inference__wrapped_model_9829
gru_1_inputN
<sequential_1_gru_1_gru_cell_1_matmul_readvariableop_resource:<P
>sequential_1_gru_1_gru_cell_1_matmul_1_readvariableop_resource:<E
3sequential_1_dense_1_matmul_readvariableop_resource:B
4sequential_1_dense_1_biasadd_readvariableop_resource:
identity??+sequential_1/dense_1/BiasAdd/ReadVariableOp?*sequential_1/dense_1/MatMul/ReadVariableOp?3sequential_1/gru_1/gru_cell_1/MatMul/ReadVariableOp?5sequential_1/gru_1/gru_cell_1/MatMul_1/ReadVariableOp?sequential_1/gru_1/whileS
sequential_1/gru_1/ShapeShapegru_1_input*
T0*
_output_shapes
:p
&sequential_1/gru_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(sequential_1/gru_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(sequential_1/gru_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 sequential_1/gru_1/strided_sliceStridedSlice!sequential_1/gru_1/Shape:output:0/sequential_1/gru_1/strided_slice/stack:output:01sequential_1/gru_1/strided_slice/stack_1:output:01sequential_1/gru_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskc
!sequential_1/gru_1/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :?
sequential_1/gru_1/zeros/packedPack)sequential_1/gru_1/strided_slice:output:0*sequential_1/gru_1/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:c
sequential_1/gru_1/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
sequential_1/gru_1/zerosFill(sequential_1/gru_1/zeros/packed:output:0'sequential_1/gru_1/zeros/Const:output:0*
T0*'
_output_shapes
:?????????v
!sequential_1/gru_1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
sequential_1/gru_1/transpose	Transposegru_1_input*sequential_1/gru_1/transpose/perm:output:0*
T0*+
_output_shapes
:?????????j
sequential_1/gru_1/Shape_1Shape sequential_1/gru_1/transpose:y:0*
T0*
_output_shapes
:r
(sequential_1/gru_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*sequential_1/gru_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*sequential_1/gru_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
"sequential_1/gru_1/strided_slice_1StridedSlice#sequential_1/gru_1/Shape_1:output:01sequential_1/gru_1/strided_slice_1/stack:output:03sequential_1/gru_1/strided_slice_1/stack_1:output:03sequential_1/gru_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masky
.sequential_1/gru_1/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
 sequential_1/gru_1/TensorArrayV2TensorListReserve7sequential_1/gru_1/TensorArrayV2/element_shape:output:0+sequential_1/gru_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
Hsequential_1/gru_1/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
:sequential_1/gru_1/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor sequential_1/gru_1/transpose:y:0Qsequential_1/gru_1/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???r
(sequential_1/gru_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*sequential_1/gru_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*sequential_1/gru_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
"sequential_1/gru_1/strided_slice_2StridedSlice sequential_1/gru_1/transpose:y:01sequential_1/gru_1/strided_slice_2/stack:output:03sequential_1/gru_1/strided_slice_2/stack_1:output:03sequential_1/gru_1/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask?
3sequential_1/gru_1/gru_cell_1/MatMul/ReadVariableOpReadVariableOp<sequential_1_gru_1_gru_cell_1_matmul_readvariableop_resource*
_output_shapes

:<*
dtype0?
$sequential_1/gru_1/gru_cell_1/MatMulMatMul+sequential_1/gru_1/strided_slice_2:output:0;sequential_1/gru_1/gru_cell_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<x
-sequential_1/gru_1/gru_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
#sequential_1/gru_1/gru_cell_1/splitSplit6sequential_1/gru_1/gru_cell_1/split/split_dim:output:0.sequential_1/gru_1/gru_cell_1/MatMul:product:0*
T0*M
_output_shapes;
9:?????????:?????????:?????????*
	num_split?
5sequential_1/gru_1/gru_cell_1/MatMul_1/ReadVariableOpReadVariableOp>sequential_1_gru_1_gru_cell_1_matmul_1_readvariableop_resource*
_output_shapes

:<*
dtype0?
&sequential_1/gru_1/gru_cell_1/MatMul_1MatMul!sequential_1/gru_1/zeros:output:0=sequential_1/gru_1/gru_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<x
#sequential_1/gru_1/gru_cell_1/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ????z
/sequential_1/gru_1/gru_cell_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
%sequential_1/gru_1/gru_cell_1/split_1SplitV0sequential_1/gru_1/gru_cell_1/MatMul_1:product:0,sequential_1/gru_1/gru_cell_1/Const:output:08sequential_1/gru_1/gru_cell_1/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:?????????:?????????:?????????*
	num_split?
!sequential_1/gru_1/gru_cell_1/addAddV2,sequential_1/gru_1/gru_cell_1/split:output:0.sequential_1/gru_1/gru_cell_1/split_1:output:0*
T0*'
_output_shapes
:??????????
%sequential_1/gru_1/gru_cell_1/SigmoidSigmoid%sequential_1/gru_1/gru_cell_1/add:z:0*
T0*'
_output_shapes
:??????????
#sequential_1/gru_1/gru_cell_1/add_1AddV2,sequential_1/gru_1/gru_cell_1/split:output:1.sequential_1/gru_1/gru_cell_1/split_1:output:1*
T0*'
_output_shapes
:??????????
'sequential_1/gru_1/gru_cell_1/Sigmoid_1Sigmoid'sequential_1/gru_1/gru_cell_1/add_1:z:0*
T0*'
_output_shapes
:??????????
!sequential_1/gru_1/gru_cell_1/mulMul+sequential_1/gru_1/gru_cell_1/Sigmoid_1:y:0.sequential_1/gru_1/gru_cell_1/split_1:output:2*
T0*'
_output_shapes
:??????????
#sequential_1/gru_1/gru_cell_1/add_2AddV2,sequential_1/gru_1/gru_cell_1/split:output:2%sequential_1/gru_1/gru_cell_1/mul:z:0*
T0*'
_output_shapes
:??????????
"sequential_1/gru_1/gru_cell_1/TanhTanh'sequential_1/gru_1/gru_cell_1/add_2:z:0*
T0*'
_output_shapes
:??????????
#sequential_1/gru_1/gru_cell_1/mul_1Mul)sequential_1/gru_1/gru_cell_1/Sigmoid:y:0!sequential_1/gru_1/zeros:output:0*
T0*'
_output_shapes
:?????????h
#sequential_1/gru_1/gru_cell_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
!sequential_1/gru_1/gru_cell_1/subSub,sequential_1/gru_1/gru_cell_1/sub/x:output:0)sequential_1/gru_1/gru_cell_1/Sigmoid:y:0*
T0*'
_output_shapes
:??????????
#sequential_1/gru_1/gru_cell_1/mul_2Mul%sequential_1/gru_1/gru_cell_1/sub:z:0&sequential_1/gru_1/gru_cell_1/Tanh:y:0*
T0*'
_output_shapes
:??????????
#sequential_1/gru_1/gru_cell_1/add_3AddV2'sequential_1/gru_1/gru_cell_1/mul_1:z:0'sequential_1/gru_1/gru_cell_1/mul_2:z:0*
T0*'
_output_shapes
:??????????
0sequential_1/gru_1/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
"sequential_1/gru_1/TensorArrayV2_1TensorListReserve9sequential_1/gru_1/TensorArrayV2_1/element_shape:output:0+sequential_1/gru_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???Y
sequential_1/gru_1/timeConst*
_output_shapes
: *
dtype0*
value	B : v
+sequential_1/gru_1/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????g
%sequential_1/gru_1/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
sequential_1/gru_1/whileWhile.sequential_1/gru_1/while/loop_counter:output:04sequential_1/gru_1/while/maximum_iterations:output:0 sequential_1/gru_1/time:output:0+sequential_1/gru_1/TensorArrayV2_1:handle:0!sequential_1/gru_1/zeros:output:0+sequential_1/gru_1/strided_slice_1:output:0Jsequential_1/gru_1/TensorArrayUnstack/TensorListFromTensor:output_handle:0<sequential_1_gru_1_gru_cell_1_matmul_readvariableop_resource>sequential_1_gru_1_gru_cell_1_matmul_1_readvariableop_resource*
T
2	*
_lower_using_switch_merge(*
_num_original_outputs	*7
_output_shapes%
#: : : : :?????????: : : : *$
_read_only_resource_inputs
*
_stateful_parallelism( *.
body&R$
"sequential_1_gru_1_while_body_9741*.
cond&R$
"sequential_1_gru_1_while_cond_9740*6
output_shapes%
#: : : : :?????????: : : : *
parallel_iterations ?
Csequential_1/gru_1/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
5sequential_1/gru_1/TensorArrayV2Stack/TensorListStackTensorListStack!sequential_1/gru_1/while:output:3Lsequential_1/gru_1/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????*
element_dtype0{
(sequential_1/gru_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????t
*sequential_1/gru_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: t
*sequential_1/gru_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
"sequential_1/gru_1/strided_slice_3StridedSlice>sequential_1/gru_1/TensorArrayV2Stack/TensorListStack:tensor:01sequential_1/gru_1/strided_slice_3/stack:output:03sequential_1/gru_1/strided_slice_3/stack_1:output:03sequential_1/gru_1/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_maskx
#sequential_1/gru_1/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
sequential_1/gru_1/transpose_1	Transpose>sequential_1/gru_1/TensorArrayV2Stack/TensorListStack:tensor:0,sequential_1/gru_1/transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????n
sequential_1/gru_1/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    ?
sequential_1/dropout_1/IdentityIdentity+sequential_1/gru_1/strided_slice_3:output:0*
T0*'
_output_shapes
:??????????
*sequential_1/dense_1/MatMul/ReadVariableOpReadVariableOp3sequential_1_dense_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
sequential_1/dense_1/MatMulMatMul(sequential_1/dropout_1/Identity:output:02sequential_1/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
+sequential_1/dense_1/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
sequential_1/dense_1/BiasAddBiasAdd%sequential_1/dense_1/MatMul:product:03sequential_1/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????t
IdentityIdentity%sequential_1/dense_1/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp,^sequential_1/dense_1/BiasAdd/ReadVariableOp+^sequential_1/dense_1/MatMul/ReadVariableOp4^sequential_1/gru_1/gru_cell_1/MatMul/ReadVariableOp6^sequential_1/gru_1/gru_cell_1/MatMul_1/ReadVariableOp^sequential_1/gru_1/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : : : 2Z
+sequential_1/dense_1/BiasAdd/ReadVariableOp+sequential_1/dense_1/BiasAdd/ReadVariableOp2X
*sequential_1/dense_1/MatMul/ReadVariableOp*sequential_1/dense_1/MatMul/ReadVariableOp2j
3sequential_1/gru_1/gru_cell_1/MatMul/ReadVariableOp3sequential_1/gru_1/gru_cell_1/MatMul/ReadVariableOp2n
5sequential_1/gru_1/gru_cell_1/MatMul_1/ReadVariableOp5sequential_1/gru_1/gru_cell_1/MatMul_1/ReadVariableOp24
sequential_1/gru_1/whilesequential_1/gru_1/while:X T
+
_output_shapes
:?????????
%
_user_specified_namegru_1_input
?	
?
B__inference_dense_1_layer_call_and_return_conditional_losses_10303

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
B__inference_dense_1_layer_call_and_return_conditional_losses_11579

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
b
D__inference_dropout_1_layer_call_and_return_conditional_losses_11548

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:?????????"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
%__inference_gru_1_layer_call_fn_10959
inputs_0
unknown:<
	unknown_0:<
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_gru_1_layer_call_and_return_conditional_losses_10128o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :??????????????????
"
_user_specified_name
inputs/0
?

?
"sequential_1_gru_1_while_cond_9740B
>sequential_1_gru_1_while_sequential_1_gru_1_while_loop_counterH
Dsequential_1_gru_1_while_sequential_1_gru_1_while_maximum_iterations(
$sequential_1_gru_1_while_placeholder*
&sequential_1_gru_1_while_placeholder_1*
&sequential_1_gru_1_while_placeholder_2D
@sequential_1_gru_1_while_less_sequential_1_gru_1_strided_slice_1X
Tsequential_1_gru_1_while_sequential_1_gru_1_while_cond_9740___redundant_placeholder0X
Tsequential_1_gru_1_while_sequential_1_gru_1_while_cond_9740___redundant_placeholder1X
Tsequential_1_gru_1_while_sequential_1_gru_1_while_cond_9740___redundant_placeholder2%
!sequential_1_gru_1_while_identity
?
sequential_1/gru_1/while/LessLess$sequential_1_gru_1_while_placeholder@sequential_1_gru_1_while_less_sequential_1_gru_1_strided_slice_1*
T0*
_output_shapes
: q
!sequential_1/gru_1/while/IdentityIdentity!sequential_1/gru_1/while/Less:z:0*
T0
*
_output_shapes
: "O
!sequential_1_gru_1_while_identity*sequential_1/gru_1/while/Identity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
): : : : :?????????: :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
:
?
?
,__inference_sequential_1_layer_call_fn_10565
gru_1_input
unknown:<
	unknown_0:<
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallgru_1_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_10541o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
+
_output_shapes
:?????????
%
_user_specified_namegru_1_input
?

?
*__inference_gru_cell_1_layer_call_fn_11591

inputs
states_0
unknown:<
	unknown_0:<
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_gru_cell_1_layer_call_and_return_conditional_losses_9893o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*=
_input_shapes,
*:?????????:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:QM
'
_output_shapes
:?????????
"
_user_specified_name
states/0
?4
?
while_body_10199
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0C
1while_gru_cell_1_matmul_readvariableop_resource_0:<E
3while_gru_cell_1_matmul_1_readvariableop_resource_0:<
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorA
/while_gru_cell_1_matmul_readvariableop_resource:<C
1while_gru_cell_1_matmul_1_readvariableop_resource:<??&while/gru_cell_1/MatMul/ReadVariableOp?(while/gru_cell_1/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0?
&while/gru_cell_1/MatMul/ReadVariableOpReadVariableOp1while_gru_cell_1_matmul_readvariableop_resource_0*
_output_shapes

:<*
dtype0?
while/gru_cell_1/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/gru_cell_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<k
 while/gru_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
while/gru_cell_1/splitSplit)while/gru_cell_1/split/split_dim:output:0!while/gru_cell_1/MatMul:product:0*
T0*M
_output_shapes;
9:?????????:?????????:?????????*
	num_split?
(while/gru_cell_1/MatMul_1/ReadVariableOpReadVariableOp3while_gru_cell_1_matmul_1_readvariableop_resource_0*
_output_shapes

:<*
dtype0?
while/gru_cell_1/MatMul_1MatMulwhile_placeholder_20while/gru_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<k
while/gru_cell_1/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ????m
"while/gru_cell_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
while/gru_cell_1/split_1SplitV#while/gru_cell_1/MatMul_1:product:0while/gru_cell_1/Const:output:0+while/gru_cell_1/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:?????????:?????????:?????????*
	num_split?
while/gru_cell_1/addAddV2while/gru_cell_1/split:output:0!while/gru_cell_1/split_1:output:0*
T0*'
_output_shapes
:?????????o
while/gru_cell_1/SigmoidSigmoidwhile/gru_cell_1/add:z:0*
T0*'
_output_shapes
:??????????
while/gru_cell_1/add_1AddV2while/gru_cell_1/split:output:1!while/gru_cell_1/split_1:output:1*
T0*'
_output_shapes
:?????????s
while/gru_cell_1/Sigmoid_1Sigmoidwhile/gru_cell_1/add_1:z:0*
T0*'
_output_shapes
:??????????
while/gru_cell_1/mulMulwhile/gru_cell_1/Sigmoid_1:y:0!while/gru_cell_1/split_1:output:2*
T0*'
_output_shapes
:??????????
while/gru_cell_1/add_2AddV2while/gru_cell_1/split:output:2while/gru_cell_1/mul:z:0*
T0*'
_output_shapes
:?????????k
while/gru_cell_1/TanhTanhwhile/gru_cell_1/add_2:z:0*
T0*'
_output_shapes
:??????????
while/gru_cell_1/mul_1Mulwhile/gru_cell_1/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:?????????[
while/gru_cell_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
while/gru_cell_1/subSubwhile/gru_cell_1/sub/x:output:0while/gru_cell_1/Sigmoid:y:0*
T0*'
_output_shapes
:??????????
while/gru_cell_1/mul_2Mulwhile/gru_cell_1/sub:z:0while/gru_cell_1/Tanh:y:0*
T0*'
_output_shapes
:??????????
while/gru_cell_1/add_3AddV2while/gru_cell_1/mul_1:z:0while/gru_cell_1/mul_2:z:0*
T0*'
_output_shapes
:??????????
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_1/add_3:z:0*
_output_shapes
: *
element_dtype0:???M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: ?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: w
while/Identity_4Identitywhile/gru_cell_1/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:??????????

while/NoOpNoOp'^while/gru_cell_1/MatMul/ReadVariableOp)^while/gru_cell_1/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "h
1while_gru_cell_1_matmul_1_readvariableop_resource3while_gru_cell_1_matmul_1_readvariableop_resource_0"d
/while_gru_cell_1_matmul_readvariableop_resource1while_gru_cell_1_matmul_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#: : : : :?????????: : : : 2P
&while/gru_cell_1/MatMul/ReadVariableOp&while/gru_cell_1/MatMul/ReadVariableOp2T
(while/gru_cell_1/MatMul_1/ReadVariableOp(while/gru_cell_1/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
: 
?
?
while_cond_11173
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_11173___redundant_placeholder03
/while_while_cond_11173___redundant_placeholder13
/while_while_cond_11173___redundant_placeholder2
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
): : : : :?????????: :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
:
?3
?
?__inference_gru_1_layer_call_and_return_conditional_losses_9964

inputs!
gru_cell_1_9894:<!
gru_cell_1_9896:<
identity??"gru_cell_1/StatefulPartitionedCall?while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask?
"gru_cell_1/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0gru_cell_1_9894gru_cell_1_9896*
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_gru_cell_1_layer_call_and_return_conditional_losses_9893n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0gru_cell_1_9894gru_cell_1_9896*
T
2	*
_lower_using_switch_merge(*
_num_original_outputs	*7
_output_shapes%
#: : : : :?????????: : : : *$
_read_only_resource_inputs
*
_stateful_parallelism( *
bodyR
while_body_9904*
condR
while_cond_9903*6
output_shapes%
#: : : : :?????????: : : : *
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:?????????s
NoOpNoOp#^gru_cell_1/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????????????: : 2H
"gru_cell_1/StatefulPartitionedCall"gru_cell_1/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
?
%__inference_gru_1_layer_call_fn_10950
inputs_0
unknown:<
	unknown_0:<
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_gru_1_layer_call_and_return_conditional_losses_9964o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :??????????????????
"
_user_specified_name
inputs/0
?
?
G__inference_sequential_1_layer_call_and_return_conditional_losses_10580
gru_1_input
gru_1_10568:<
gru_1_10570:<
dense_1_10574:
dense_1_10576:
identity??dense_1/StatefulPartitionedCall?gru_1/StatefulPartitionedCall?
gru_1/StatefulPartitionedCallStatefulPartitionedCallgru_1_inputgru_1_10568gru_1_10570*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_gru_1_layer_call_and_return_conditional_losses_10280?
dropout_1/PartitionedCallPartitionedCall&gru_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_10291?
dense_1/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0dense_1_10574dense_1_10576*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_10303w
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp ^dense_1/StatefulPartitionedCall^gru_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : : : 2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2>
gru_1/StatefulPartitionedCallgru_1/StatefulPartitionedCall:X T
+
_output_shapes
:?????????
%
_user_specified_namegru_1_input
?/
?
__inference__traced_save_11749
file_prefix-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop6
2savev2_gru_1_gru_cell_1_kernel_read_readvariableop@
<savev2_gru_1_gru_cell_1_recurrent_kernel_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop4
0savev2_adam_dense_1_kernel_m_read_readvariableop2
.savev2_adam_dense_1_bias_m_read_readvariableop=
9savev2_adam_gru_1_gru_cell_1_kernel_m_read_readvariableopG
Csavev2_adam_gru_1_gru_cell_1_recurrent_kernel_m_read_readvariableop4
0savev2_adam_dense_1_kernel_v_read_readvariableop2
.savev2_adam_dense_1_bias_v_read_readvariableop=
9savev2_adam_gru_1_gru_cell_1_kernel_v_read_readvariableopG
Csavev2_adam_gru_1_gru_cell_1_recurrent_kernel_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : ?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ?	
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?	
value?	B?	B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*;
value2B0B B B B B B B B B B B B B B B B B B B B ?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop2savev2_gru_1_gru_cell_1_kernel_read_readvariableop<savev2_gru_1_gru_cell_1_recurrent_kernel_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop0savev2_adam_dense_1_kernel_m_read_readvariableop.savev2_adam_dense_1_bias_m_read_readvariableop9savev2_adam_gru_1_gru_cell_1_kernel_m_read_readvariableopCsavev2_adam_gru_1_gru_cell_1_recurrent_kernel_m_read_readvariableop0savev2_adam_dense_1_kernel_v_read_readvariableop.savev2_adam_dense_1_bias_v_read_readvariableop9savev2_adam_gru_1_gru_cell_1_kernel_v_read_readvariableopCsavev2_adam_gru_1_gru_cell_1_recurrent_kernel_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *"
dtypes
2	?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*?
_input_shapes?
~: ::: : : : : :<:<: : :::<:<:::<:<: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:<:$	 

_output_shapes

:<:


_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:<:$ 

_output_shapes

:<:$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:<:$ 

_output_shapes

:<:

_output_shapes
: 
?	
c
D__inference_dropout_1_layer_call_and_return_conditional_losses_11560

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:?????????C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:?????????Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
while_cond_10198
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_10198___redundant_placeholder03
/while_while_cond_10198___redundant_placeholder13
/while_while_cond_10198___redundant_placeholder2
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
): : : : :?????????: :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
:
?
?
while_cond_10067
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_10067___redundant_placeholder03
/while_while_cond_10067___redundant_placeholder13
/while_while_cond_10067___redundant_placeholder2
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
): : : : :?????????: :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
:
?3
?
@__inference_gru_1_layer_call_and_return_conditional_losses_10128

inputs"
gru_cell_1_10058:<"
gru_cell_1_10060:<
identity??"gru_cell_1/StatefulPartitionedCall?while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask?
"gru_cell_1/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0gru_cell_1_10058gru_cell_1_10060*
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_gru_cell_1_layer_call_and_return_conditional_losses_10020n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0gru_cell_1_10058gru_cell_1_10060*
T
2	*
_lower_using_switch_merge(*
_num_original_outputs	*7
_output_shapes%
#: : : : :?????????: : : : *$
_read_only_resource_inputs
*
_stateful_parallelism( *
bodyR
while_body_10068*
condR
while_cond_10067*6
output_shapes%
#: : : : :?????????: : : : *
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:?????????s
NoOpNoOp#^gru_cell_1/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????????????: : 2H
"gru_cell_1/StatefulPartitionedCall"gru_cell_1/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?F
?
@__inference_gru_1_layer_call_and_return_conditional_losses_11533

inputs;
)gru_cell_1_matmul_readvariableop_resource:<=
+gru_cell_1_matmul_1_readvariableop_resource:<
identity?? gru_cell_1/MatMul/ReadVariableOp?"gru_cell_1/MatMul_1/ReadVariableOp?while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:?????????D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask?
 gru_cell_1/MatMul/ReadVariableOpReadVariableOp)gru_cell_1_matmul_readvariableop_resource*
_output_shapes

:<*
dtype0?
gru_cell_1/MatMulMatMulstrided_slice_2:output:0(gru_cell_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<e
gru_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
gru_cell_1/splitSplit#gru_cell_1/split/split_dim:output:0gru_cell_1/MatMul:product:0*
T0*M
_output_shapes;
9:?????????:?????????:?????????*
	num_split?
"gru_cell_1/MatMul_1/ReadVariableOpReadVariableOp+gru_cell_1_matmul_1_readvariableop_resource*
_output_shapes

:<*
dtype0?
gru_cell_1/MatMul_1MatMulzeros:output:0*gru_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<e
gru_cell_1/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ????g
gru_cell_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
gru_cell_1/split_1SplitVgru_cell_1/MatMul_1:product:0gru_cell_1/Const:output:0%gru_cell_1/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:?????????:?????????:?????????*
	num_split?
gru_cell_1/addAddV2gru_cell_1/split:output:0gru_cell_1/split_1:output:0*
T0*'
_output_shapes
:?????????c
gru_cell_1/SigmoidSigmoidgru_cell_1/add:z:0*
T0*'
_output_shapes
:??????????
gru_cell_1/add_1AddV2gru_cell_1/split:output:1gru_cell_1/split_1:output:1*
T0*'
_output_shapes
:?????????g
gru_cell_1/Sigmoid_1Sigmoidgru_cell_1/add_1:z:0*
T0*'
_output_shapes
:?????????~
gru_cell_1/mulMulgru_cell_1/Sigmoid_1:y:0gru_cell_1/split_1:output:2*
T0*'
_output_shapes
:?????????z
gru_cell_1/add_2AddV2gru_cell_1/split:output:2gru_cell_1/mul:z:0*
T0*'
_output_shapes
:?????????_
gru_cell_1/TanhTanhgru_cell_1/add_2:z:0*
T0*'
_output_shapes
:?????????q
gru_cell_1/mul_1Mulgru_cell_1/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:?????????U
gru_cell_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??z
gru_cell_1/subSubgru_cell_1/sub/x:output:0gru_cell_1/Sigmoid:y:0*
T0*'
_output_shapes
:?????????r
gru_cell_1/mul_2Mulgru_cell_1/sub:z:0gru_cell_1/Tanh:y:0*
T0*'
_output_shapes
:?????????w
gru_cell_1/add_3AddV2gru_cell_1/mul_1:z:0gru_cell_1/mul_2:z:0*
T0*'
_output_shapes
:?????????n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)gru_cell_1_matmul_readvariableop_resource+gru_cell_1_matmul_1_readvariableop_resource*
T
2	*
_lower_using_switch_merge(*
_num_original_outputs	*7
_output_shapes%
#: : : : :?????????: : : : *$
_read_only_resource_inputs
*
_stateful_parallelism( *
bodyR
while_body_11452*
condR
while_cond_11451*6
output_shapes%
#: : : : :?????????: : : : *
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp!^gru_cell_1/MatMul/ReadVariableOp#^gru_cell_1/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : 2D
 gru_cell_1/MatMul/ReadVariableOp gru_cell_1/MatMul/ReadVariableOp2H
"gru_cell_1/MatMul_1/ReadVariableOp"gru_cell_1/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?]
?
G__inference_sequential_1_layer_call_and_return_conditional_losses_10941

inputsA
/gru_1_gru_cell_1_matmul_readvariableop_resource:<C
1gru_1_gru_cell_1_matmul_1_readvariableop_resource:<8
&dense_1_matmul_readvariableop_resource:5
'dense_1_biasadd_readvariableop_resource:
identity??dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?&gru_1/gru_cell_1/MatMul/ReadVariableOp?(gru_1/gru_cell_1/MatMul_1/ReadVariableOp?gru_1/whileA
gru_1/ShapeShapeinputs*
T0*
_output_shapes
:c
gru_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: e
gru_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:e
gru_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
gru_1/strided_sliceStridedSlicegru_1/Shape:output:0"gru_1/strided_slice/stack:output:0$gru_1/strided_slice/stack_1:output:0$gru_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskV
gru_1/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :?
gru_1/zeros/packedPackgru_1/strided_slice:output:0gru_1/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:V
gru_1/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ~
gru_1/zerosFillgru_1/zeros/packed:output:0gru_1/zeros/Const:output:0*
T0*'
_output_shapes
:?????????i
gru_1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          y
gru_1/transpose	Transposeinputsgru_1/transpose/perm:output:0*
T0*+
_output_shapes
:?????????P
gru_1/Shape_1Shapegru_1/transpose:y:0*
T0*
_output_shapes
:e
gru_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: g
gru_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
gru_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
gru_1/strided_slice_1StridedSlicegru_1/Shape_1:output:0$gru_1/strided_slice_1/stack:output:0&gru_1/strided_slice_1/stack_1:output:0&gru_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskl
!gru_1/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
gru_1/TensorArrayV2TensorListReserve*gru_1/TensorArrayV2/element_shape:output:0gru_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
;gru_1/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
-gru_1/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorgru_1/transpose:y:0Dgru_1/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???e
gru_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: g
gru_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
gru_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
gru_1/strided_slice_2StridedSlicegru_1/transpose:y:0$gru_1/strided_slice_2/stack:output:0&gru_1/strided_slice_2/stack_1:output:0&gru_1/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask?
&gru_1/gru_cell_1/MatMul/ReadVariableOpReadVariableOp/gru_1_gru_cell_1_matmul_readvariableop_resource*
_output_shapes

:<*
dtype0?
gru_1/gru_cell_1/MatMulMatMulgru_1/strided_slice_2:output:0.gru_1/gru_cell_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<k
 gru_1/gru_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
gru_1/gru_cell_1/splitSplit)gru_1/gru_cell_1/split/split_dim:output:0!gru_1/gru_cell_1/MatMul:product:0*
T0*M
_output_shapes;
9:?????????:?????????:?????????*
	num_split?
(gru_1/gru_cell_1/MatMul_1/ReadVariableOpReadVariableOp1gru_1_gru_cell_1_matmul_1_readvariableop_resource*
_output_shapes

:<*
dtype0?
gru_1/gru_cell_1/MatMul_1MatMulgru_1/zeros:output:00gru_1/gru_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<k
gru_1/gru_cell_1/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ????m
"gru_1/gru_cell_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
gru_1/gru_cell_1/split_1SplitV#gru_1/gru_cell_1/MatMul_1:product:0gru_1/gru_cell_1/Const:output:0+gru_1/gru_cell_1/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:?????????:?????????:?????????*
	num_split?
gru_1/gru_cell_1/addAddV2gru_1/gru_cell_1/split:output:0!gru_1/gru_cell_1/split_1:output:0*
T0*'
_output_shapes
:?????????o
gru_1/gru_cell_1/SigmoidSigmoidgru_1/gru_cell_1/add:z:0*
T0*'
_output_shapes
:??????????
gru_1/gru_cell_1/add_1AddV2gru_1/gru_cell_1/split:output:1!gru_1/gru_cell_1/split_1:output:1*
T0*'
_output_shapes
:?????????s
gru_1/gru_cell_1/Sigmoid_1Sigmoidgru_1/gru_cell_1/add_1:z:0*
T0*'
_output_shapes
:??????????
gru_1/gru_cell_1/mulMulgru_1/gru_cell_1/Sigmoid_1:y:0!gru_1/gru_cell_1/split_1:output:2*
T0*'
_output_shapes
:??????????
gru_1/gru_cell_1/add_2AddV2gru_1/gru_cell_1/split:output:2gru_1/gru_cell_1/mul:z:0*
T0*'
_output_shapes
:?????????k
gru_1/gru_cell_1/TanhTanhgru_1/gru_cell_1/add_2:z:0*
T0*'
_output_shapes
:??????????
gru_1/gru_cell_1/mul_1Mulgru_1/gru_cell_1/Sigmoid:y:0gru_1/zeros:output:0*
T0*'
_output_shapes
:?????????[
gru_1/gru_cell_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
gru_1/gru_cell_1/subSubgru_1/gru_cell_1/sub/x:output:0gru_1/gru_cell_1/Sigmoid:y:0*
T0*'
_output_shapes
:??????????
gru_1/gru_cell_1/mul_2Mulgru_1/gru_cell_1/sub:z:0gru_1/gru_cell_1/Tanh:y:0*
T0*'
_output_shapes
:??????????
gru_1/gru_cell_1/add_3AddV2gru_1/gru_cell_1/mul_1:z:0gru_1/gru_cell_1/mul_2:z:0*
T0*'
_output_shapes
:?????????t
#gru_1/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
gru_1/TensorArrayV2_1TensorListReserve,gru_1/TensorArrayV2_1/element_shape:output:0gru_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???L

gru_1/timeConst*
_output_shapes
: *
dtype0*
value	B : i
gru_1/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????Z
gru_1/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
gru_1/whileWhile!gru_1/while/loop_counter:output:0'gru_1/while/maximum_iterations:output:0gru_1/time:output:0gru_1/TensorArrayV2_1:handle:0gru_1/zeros:output:0gru_1/strided_slice_1:output:0=gru_1/TensorArrayUnstack/TensorListFromTensor:output_handle:0/gru_1_gru_cell_1_matmul_readvariableop_resource1gru_1_gru_cell_1_matmul_1_readvariableop_resource*
T
2	*
_lower_using_switch_merge(*
_num_original_outputs	*7
_output_shapes%
#: : : : :?????????: : : : *$
_read_only_resource_inputs
*
_stateful_parallelism( *"
bodyR
gru_1_while_body_10846*"
condR
gru_1_while_cond_10845*6
output_shapes%
#: : : : :?????????: : : : *
parallel_iterations ?
6gru_1/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
(gru_1/TensorArrayV2Stack/TensorListStackTensorListStackgru_1/while:output:3?gru_1/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????*
element_dtype0n
gru_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????g
gru_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: g
gru_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
gru_1/strided_slice_3StridedSlice1gru_1/TensorArrayV2Stack/TensorListStack:tensor:0$gru_1/strided_slice_3/stack:output:0&gru_1/strided_slice_3/stack_1:output:0&gru_1/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_maskk
gru_1/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
gru_1/transpose_1	Transpose1gru_1/TensorArrayV2Stack/TensorListStack:tensor:0gru_1/transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????a
gru_1/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    \
dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8???
dropout_1/dropout/MulMulgru_1/strided_slice_3:output:0 dropout_1/dropout/Const:output:0*
T0*'
_output_shapes
:?????????e
dropout_1/dropout/ShapeShapegru_1/strided_slice_3:output:0*
T0*
_output_shapes
:?
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0e
 dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
dropout_1/dropout/GreaterEqualGreaterEqual7dropout_1/dropout/random_uniform/RandomUniform:output:0)dropout_1/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:??????????
dropout_1/dropout/CastCast"dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:??????????
dropout_1/dropout/Mul_1Muldropout_1/dropout/Mul:z:0dropout_1/dropout/Cast:y:0*
T0*'
_output_shapes
:??????????
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
dense_1/MatMulMatMuldropout_1/dropout/Mul_1:z:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????g
IdentityIdentitydense_1/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp'^gru_1/gru_cell_1/MatMul/ReadVariableOp)^gru_1/gru_cell_1/MatMul_1/ReadVariableOp^gru_1/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : : : 2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2P
&gru_1/gru_cell_1/MatMul/ReadVariableOp&gru_1/gru_cell_1/MatMul/ReadVariableOp2T
(gru_1/gru_cell_1/MatMul_1/ReadVariableOp(gru_1/gru_cell_1/MatMul_1/ReadVariableOp2
gru_1/whilegru_1/while:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
%__inference_gru_1_layer_call_fn_10968

inputs
unknown:<
	unknown_0:<
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_gru_1_layer_call_and_return_conditional_losses_10280o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
E
)__inference_dropout_1_layer_call_fn_11538

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_10291`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
G__inference_sequential_1_layer_call_and_return_conditional_losses_10595
gru_1_input
gru_1_10583:<
gru_1_10585:<
dense_1_10589:
dense_1_10591:
identity??dense_1/StatefulPartitionedCall?!dropout_1/StatefulPartitionedCall?gru_1/StatefulPartitionedCall?
gru_1/StatefulPartitionedCallStatefulPartitionedCallgru_1_inputgru_1_10583gru_1_10585*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_gru_1_layer_call_and_return_conditional_losses_10504?
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall&gru_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_10351?
dense_1/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0dense_1_10589dense_1_10591*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_10303w
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp ^dense_1/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall^gru_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : : : 2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2>
gru_1/StatefulPartitionedCallgru_1/StatefulPartitionedCall:X T
+
_output_shapes
:?????????
%
_user_specified_namegru_1_input
?
?
,__inference_sequential_1_layer_call_fn_10629

inputs
unknown:<
	unknown_0:<
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_10310o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
gru_1_while_cond_10699(
$gru_1_while_gru_1_while_loop_counter.
*gru_1_while_gru_1_while_maximum_iterations
gru_1_while_placeholder
gru_1_while_placeholder_1
gru_1_while_placeholder_2*
&gru_1_while_less_gru_1_strided_slice_1?
;gru_1_while_gru_1_while_cond_10699___redundant_placeholder0?
;gru_1_while_gru_1_while_cond_10699___redundant_placeholder1?
;gru_1_while_gru_1_while_cond_10699___redundant_placeholder2
gru_1_while_identity
z
gru_1/while/LessLessgru_1_while_placeholder&gru_1_while_less_gru_1_strided_slice_1*
T0*
_output_shapes
: W
gru_1/while/IdentityIdentitygru_1/while/Less:z:0*
T0
*
_output_shapes
: "5
gru_1_while_identitygru_1/while/Identity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
): : : : :?????????: :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
:
?
?
while_cond_11312
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_11312___redundant_placeholder03
/while_while_cond_11312___redundant_placeholder13
/while_while_cond_11312___redundant_placeholder2
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
): : : : :?????????: :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
:
?
b
D__inference_dropout_1_layer_call_and_return_conditional_losses_10291

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:?????????"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
G__inference_sequential_1_layer_call_and_return_conditional_losses_10310

inputs
gru_1_10281:<
gru_1_10283:<
dense_1_10304:
dense_1_10306:
identity??dense_1/StatefulPartitionedCall?gru_1/StatefulPartitionedCall?
gru_1/StatefulPartitionedCallStatefulPartitionedCallinputsgru_1_10281gru_1_10283*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_gru_1_layer_call_and_return_conditional_losses_10280?
dropout_1/PartitionedCallPartitionedCall&gru_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_10291?
dense_1/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0dense_1_10304dense_1_10306*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_10303w
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp ^dense_1/StatefulPartitionedCall^gru_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : : : 2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2>
gru_1/StatefulPartitionedCallgru_1/StatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?;
?
gru_1_while_body_10700(
$gru_1_while_gru_1_while_loop_counter.
*gru_1_while_gru_1_while_maximum_iterations
gru_1_while_placeholder
gru_1_while_placeholder_1
gru_1_while_placeholder_2'
#gru_1_while_gru_1_strided_slice_1_0c
_gru_1_while_tensorarrayv2read_tensorlistgetitem_gru_1_tensorarrayunstack_tensorlistfromtensor_0I
7gru_1_while_gru_cell_1_matmul_readvariableop_resource_0:<K
9gru_1_while_gru_cell_1_matmul_1_readvariableop_resource_0:<
gru_1_while_identity
gru_1_while_identity_1
gru_1_while_identity_2
gru_1_while_identity_3
gru_1_while_identity_4%
!gru_1_while_gru_1_strided_slice_1a
]gru_1_while_tensorarrayv2read_tensorlistgetitem_gru_1_tensorarrayunstack_tensorlistfromtensorG
5gru_1_while_gru_cell_1_matmul_readvariableop_resource:<I
7gru_1_while_gru_cell_1_matmul_1_readvariableop_resource:<??,gru_1/while/gru_cell_1/MatMul/ReadVariableOp?.gru_1/while/gru_cell_1/MatMul_1/ReadVariableOp?
=gru_1/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
/gru_1/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem_gru_1_while_tensorarrayv2read_tensorlistgetitem_gru_1_tensorarrayunstack_tensorlistfromtensor_0gru_1_while_placeholderFgru_1/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0?
,gru_1/while/gru_cell_1/MatMul/ReadVariableOpReadVariableOp7gru_1_while_gru_cell_1_matmul_readvariableop_resource_0*
_output_shapes

:<*
dtype0?
gru_1/while/gru_cell_1/MatMulMatMul6gru_1/while/TensorArrayV2Read/TensorListGetItem:item:04gru_1/while/gru_cell_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<q
&gru_1/while/gru_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
gru_1/while/gru_cell_1/splitSplit/gru_1/while/gru_cell_1/split/split_dim:output:0'gru_1/while/gru_cell_1/MatMul:product:0*
T0*M
_output_shapes;
9:?????????:?????????:?????????*
	num_split?
.gru_1/while/gru_cell_1/MatMul_1/ReadVariableOpReadVariableOp9gru_1_while_gru_cell_1_matmul_1_readvariableop_resource_0*
_output_shapes

:<*
dtype0?
gru_1/while/gru_cell_1/MatMul_1MatMulgru_1_while_placeholder_26gru_1/while/gru_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<q
gru_1/while/gru_cell_1/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ????s
(gru_1/while/gru_cell_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
gru_1/while/gru_cell_1/split_1SplitV)gru_1/while/gru_cell_1/MatMul_1:product:0%gru_1/while/gru_cell_1/Const:output:01gru_1/while/gru_cell_1/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:?????????:?????????:?????????*
	num_split?
gru_1/while/gru_cell_1/addAddV2%gru_1/while/gru_cell_1/split:output:0'gru_1/while/gru_cell_1/split_1:output:0*
T0*'
_output_shapes
:?????????{
gru_1/while/gru_cell_1/SigmoidSigmoidgru_1/while/gru_cell_1/add:z:0*
T0*'
_output_shapes
:??????????
gru_1/while/gru_cell_1/add_1AddV2%gru_1/while/gru_cell_1/split:output:1'gru_1/while/gru_cell_1/split_1:output:1*
T0*'
_output_shapes
:?????????
 gru_1/while/gru_cell_1/Sigmoid_1Sigmoid gru_1/while/gru_cell_1/add_1:z:0*
T0*'
_output_shapes
:??????????
gru_1/while/gru_cell_1/mulMul$gru_1/while/gru_cell_1/Sigmoid_1:y:0'gru_1/while/gru_cell_1/split_1:output:2*
T0*'
_output_shapes
:??????????
gru_1/while/gru_cell_1/add_2AddV2%gru_1/while/gru_cell_1/split:output:2gru_1/while/gru_cell_1/mul:z:0*
T0*'
_output_shapes
:?????????w
gru_1/while/gru_cell_1/TanhTanh gru_1/while/gru_cell_1/add_2:z:0*
T0*'
_output_shapes
:??????????
gru_1/while/gru_cell_1/mul_1Mul"gru_1/while/gru_cell_1/Sigmoid:y:0gru_1_while_placeholder_2*
T0*'
_output_shapes
:?????????a
gru_1/while/gru_cell_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
gru_1/while/gru_cell_1/subSub%gru_1/while/gru_cell_1/sub/x:output:0"gru_1/while/gru_cell_1/Sigmoid:y:0*
T0*'
_output_shapes
:??????????
gru_1/while/gru_cell_1/mul_2Mulgru_1/while/gru_cell_1/sub:z:0gru_1/while/gru_cell_1/Tanh:y:0*
T0*'
_output_shapes
:??????????
gru_1/while/gru_cell_1/add_3AddV2 gru_1/while/gru_cell_1/mul_1:z:0 gru_1/while/gru_cell_1/mul_2:z:0*
T0*'
_output_shapes
:??????????
0gru_1/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemgru_1_while_placeholder_1gru_1_while_placeholder gru_1/while/gru_cell_1/add_3:z:0*
_output_shapes
: *
element_dtype0:???S
gru_1/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :n
gru_1/while/addAddV2gru_1_while_placeholdergru_1/while/add/y:output:0*
T0*
_output_shapes
: U
gru_1/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
gru_1/while/add_1AddV2$gru_1_while_gru_1_while_loop_countergru_1/while/add_1/y:output:0*
T0*
_output_shapes
: k
gru_1/while/IdentityIdentitygru_1/while/add_1:z:0^gru_1/while/NoOp*
T0*
_output_shapes
: ?
gru_1/while/Identity_1Identity*gru_1_while_gru_1_while_maximum_iterations^gru_1/while/NoOp*
T0*
_output_shapes
: k
gru_1/while/Identity_2Identitygru_1/while/add:z:0^gru_1/while/NoOp*
T0*
_output_shapes
: ?
gru_1/while/Identity_3Identity@gru_1/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^gru_1/while/NoOp*
T0*
_output_shapes
: ?
gru_1/while/Identity_4Identity gru_1/while/gru_cell_1/add_3:z:0^gru_1/while/NoOp*
T0*'
_output_shapes
:??????????
gru_1/while/NoOpNoOp-^gru_1/while/gru_cell_1/MatMul/ReadVariableOp/^gru_1/while/gru_cell_1/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "H
!gru_1_while_gru_1_strided_slice_1#gru_1_while_gru_1_strided_slice_1_0"t
7gru_1_while_gru_cell_1_matmul_1_readvariableop_resource9gru_1_while_gru_cell_1_matmul_1_readvariableop_resource_0"p
5gru_1_while_gru_cell_1_matmul_readvariableop_resource7gru_1_while_gru_cell_1_matmul_readvariableop_resource_0"5
gru_1_while_identitygru_1/while/Identity:output:0"9
gru_1_while_identity_1gru_1/while/Identity_1:output:0"9
gru_1_while_identity_2gru_1/while/Identity_2:output:0"9
gru_1_while_identity_3gru_1/while/Identity_3:output:0"9
gru_1_while_identity_4gru_1/while/Identity_4:output:0"?
]gru_1_while_tensorarrayv2read_tensorlistgetitem_gru_1_tensorarrayunstack_tensorlistfromtensor_gru_1_while_tensorarrayv2read_tensorlistgetitem_gru_1_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#: : : : :?????????: : : : 2\
,gru_1/while/gru_cell_1/MatMul/ReadVariableOp,gru_1/while/gru_cell_1/MatMul/ReadVariableOp2`
.gru_1/while/gru_cell_1/MatMul_1/ReadVariableOp.gru_1/while/gru_cell_1/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
: 
?
?
E__inference_gru_cell_1_layer_call_and_return_conditional_losses_11669

inputs
states_00
matmul_readvariableop_resource:<2
 matmul_1_readvariableop_resource:<
identity

identity_1??MatMul/ReadVariableOp?MatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:<*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<Z
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
splitSplitsplit/split_dim:output:0MatMul:product:0*
T0*M
_output_shapes;
9:?????????:?????????:?????????*
	num_splitx
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:<*
dtype0o
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<Z
ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ????\
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
split_1SplitVMatMul_1:product:0Const:output:0split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:?????????:?????????:?????????*
	num_split`
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:?????????M
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:?????????b
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:?????????Q
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:?????????]
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:?????????Y
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:?????????I
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:?????????U
mul_1MulSigmoid:y:0states_0*
T0*'
_output_shapes
:?????????J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:?????????Q
mul_2Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:?????????V
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:?????????X
IdentityIdentity	add_3:z:0^NoOp*
T0*'
_output_shapes
:?????????Z

Identity_1Identity	add_3:z:0^NoOp*
T0*'
_output_shapes
:?????????x
NoOpNoOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*=
_input_shapes,
*:?????????:?????????: : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:QM
'
_output_shapes
:?????????
"
_user_specified_name
states/0
?	
c
D__inference_dropout_1_layer_call_and_return_conditional_losses_10351

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:?????????C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:?????????Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?4
?
while_body_11313
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0C
1while_gru_cell_1_matmul_readvariableop_resource_0:<E
3while_gru_cell_1_matmul_1_readvariableop_resource_0:<
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorA
/while_gru_cell_1_matmul_readvariableop_resource:<C
1while_gru_cell_1_matmul_1_readvariableop_resource:<??&while/gru_cell_1/MatMul/ReadVariableOp?(while/gru_cell_1/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0?
&while/gru_cell_1/MatMul/ReadVariableOpReadVariableOp1while_gru_cell_1_matmul_readvariableop_resource_0*
_output_shapes

:<*
dtype0?
while/gru_cell_1/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/gru_cell_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<k
 while/gru_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
while/gru_cell_1/splitSplit)while/gru_cell_1/split/split_dim:output:0!while/gru_cell_1/MatMul:product:0*
T0*M
_output_shapes;
9:?????????:?????????:?????????*
	num_split?
(while/gru_cell_1/MatMul_1/ReadVariableOpReadVariableOp3while_gru_cell_1_matmul_1_readvariableop_resource_0*
_output_shapes

:<*
dtype0?
while/gru_cell_1/MatMul_1MatMulwhile_placeholder_20while/gru_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<k
while/gru_cell_1/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ????m
"while/gru_cell_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
while/gru_cell_1/split_1SplitV#while/gru_cell_1/MatMul_1:product:0while/gru_cell_1/Const:output:0+while/gru_cell_1/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:?????????:?????????:?????????*
	num_split?
while/gru_cell_1/addAddV2while/gru_cell_1/split:output:0!while/gru_cell_1/split_1:output:0*
T0*'
_output_shapes
:?????????o
while/gru_cell_1/SigmoidSigmoidwhile/gru_cell_1/add:z:0*
T0*'
_output_shapes
:??????????
while/gru_cell_1/add_1AddV2while/gru_cell_1/split:output:1!while/gru_cell_1/split_1:output:1*
T0*'
_output_shapes
:?????????s
while/gru_cell_1/Sigmoid_1Sigmoidwhile/gru_cell_1/add_1:z:0*
T0*'
_output_shapes
:??????????
while/gru_cell_1/mulMulwhile/gru_cell_1/Sigmoid_1:y:0!while/gru_cell_1/split_1:output:2*
T0*'
_output_shapes
:??????????
while/gru_cell_1/add_2AddV2while/gru_cell_1/split:output:2while/gru_cell_1/mul:z:0*
T0*'
_output_shapes
:?????????k
while/gru_cell_1/TanhTanhwhile/gru_cell_1/add_2:z:0*
T0*'
_output_shapes
:??????????
while/gru_cell_1/mul_1Mulwhile/gru_cell_1/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:?????????[
while/gru_cell_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
while/gru_cell_1/subSubwhile/gru_cell_1/sub/x:output:0while/gru_cell_1/Sigmoid:y:0*
T0*'
_output_shapes
:??????????
while/gru_cell_1/mul_2Mulwhile/gru_cell_1/sub:z:0while/gru_cell_1/Tanh:y:0*
T0*'
_output_shapes
:??????????
while/gru_cell_1/add_3AddV2while/gru_cell_1/mul_1:z:0while/gru_cell_1/mul_2:z:0*
T0*'
_output_shapes
:??????????
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_1/add_3:z:0*
_output_shapes
: *
element_dtype0:???M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: ?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: w
while/Identity_4Identitywhile/gru_cell_1/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:??????????

while/NoOpNoOp'^while/gru_cell_1/MatMul/ReadVariableOp)^while/gru_cell_1/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "h
1while_gru_cell_1_matmul_1_readvariableop_resource3while_gru_cell_1_matmul_1_readvariableop_resource_0"d
/while_gru_cell_1_matmul_readvariableop_resource1while_gru_cell_1_matmul_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#: : : : :?????????: : : : 2P
&while/gru_cell_1/MatMul/ReadVariableOp&while/gru_cell_1/MatMul/ReadVariableOp2T
(while/gru_cell_1/MatMul_1/ReadVariableOp(while/gru_cell_1/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
: 
?
?
,__inference_sequential_1_layer_call_fn_10321
gru_1_input
unknown:<
	unknown_0:<
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallgru_1_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_10310o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
+
_output_shapes
:?????????
%
_user_specified_namegru_1_input
?
?
gru_1_while_cond_10845(
$gru_1_while_gru_1_while_loop_counter.
*gru_1_while_gru_1_while_maximum_iterations
gru_1_while_placeholder
gru_1_while_placeholder_1
gru_1_while_placeholder_2*
&gru_1_while_less_gru_1_strided_slice_1?
;gru_1_while_gru_1_while_cond_10845___redundant_placeholder0?
;gru_1_while_gru_1_while_cond_10845___redundant_placeholder1?
;gru_1_while_gru_1_while_cond_10845___redundant_placeholder2
gru_1_while_identity
z
gru_1/while/LessLessgru_1_while_placeholder&gru_1_while_less_gru_1_strided_slice_1*
T0*
_output_shapes
: W
gru_1/while/IdentityIdentitygru_1/while/Less:z:0*
T0
*
_output_shapes
: "5
gru_1_while_identitygru_1/while/Identity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
): : : : :?????????: :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
:
?
?
E__inference_gru_cell_1_layer_call_and_return_conditional_losses_11636

inputs
states_00
matmul_readvariableop_resource:<2
 matmul_1_readvariableop_resource:<
identity

identity_1??MatMul/ReadVariableOp?MatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:<*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<Z
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
splitSplitsplit/split_dim:output:0MatMul:product:0*
T0*M
_output_shapes;
9:?????????:?????????:?????????*
	num_splitx
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:<*
dtype0o
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<Z
ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ????\
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
split_1SplitVMatMul_1:product:0Const:output:0split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:?????????:?????????:?????????*
	num_split`
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:?????????M
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:?????????b
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:?????????Q
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:?????????]
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:?????????Y
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:?????????I
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:?????????U
mul_1MulSigmoid:y:0states_0*
T0*'
_output_shapes
:?????????J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:?????????Q
mul_2Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:?????????V
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:?????????X
IdentityIdentity	add_3:z:0^NoOp*
T0*'
_output_shapes
:?????????Z

Identity_1Identity	add_3:z:0^NoOp*
T0*'
_output_shapes
:?????????x
NoOpNoOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*=
_input_shapes,
*:?????????:?????????: : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:QM
'
_output_shapes
:?????????
"
_user_specified_name
states/0
?J
?

"sequential_1_gru_1_while_body_9741B
>sequential_1_gru_1_while_sequential_1_gru_1_while_loop_counterH
Dsequential_1_gru_1_while_sequential_1_gru_1_while_maximum_iterations(
$sequential_1_gru_1_while_placeholder*
&sequential_1_gru_1_while_placeholder_1*
&sequential_1_gru_1_while_placeholder_2A
=sequential_1_gru_1_while_sequential_1_gru_1_strided_slice_1_0}
ysequential_1_gru_1_while_tensorarrayv2read_tensorlistgetitem_sequential_1_gru_1_tensorarrayunstack_tensorlistfromtensor_0V
Dsequential_1_gru_1_while_gru_cell_1_matmul_readvariableop_resource_0:<X
Fsequential_1_gru_1_while_gru_cell_1_matmul_1_readvariableop_resource_0:<%
!sequential_1_gru_1_while_identity'
#sequential_1_gru_1_while_identity_1'
#sequential_1_gru_1_while_identity_2'
#sequential_1_gru_1_while_identity_3'
#sequential_1_gru_1_while_identity_4?
;sequential_1_gru_1_while_sequential_1_gru_1_strided_slice_1{
wsequential_1_gru_1_while_tensorarrayv2read_tensorlistgetitem_sequential_1_gru_1_tensorarrayunstack_tensorlistfromtensorT
Bsequential_1_gru_1_while_gru_cell_1_matmul_readvariableop_resource:<V
Dsequential_1_gru_1_while_gru_cell_1_matmul_1_readvariableop_resource:<??9sequential_1/gru_1/while/gru_cell_1/MatMul/ReadVariableOp?;sequential_1/gru_1/while/gru_cell_1/MatMul_1/ReadVariableOp?
Jsequential_1/gru_1/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
<sequential_1/gru_1/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemysequential_1_gru_1_while_tensorarrayv2read_tensorlistgetitem_sequential_1_gru_1_tensorarrayunstack_tensorlistfromtensor_0$sequential_1_gru_1_while_placeholderSsequential_1/gru_1/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0?
9sequential_1/gru_1/while/gru_cell_1/MatMul/ReadVariableOpReadVariableOpDsequential_1_gru_1_while_gru_cell_1_matmul_readvariableop_resource_0*
_output_shapes

:<*
dtype0?
*sequential_1/gru_1/while/gru_cell_1/MatMulMatMulCsequential_1/gru_1/while/TensorArrayV2Read/TensorListGetItem:item:0Asequential_1/gru_1/while/gru_cell_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<~
3sequential_1/gru_1/while/gru_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
)sequential_1/gru_1/while/gru_cell_1/splitSplit<sequential_1/gru_1/while/gru_cell_1/split/split_dim:output:04sequential_1/gru_1/while/gru_cell_1/MatMul:product:0*
T0*M
_output_shapes;
9:?????????:?????????:?????????*
	num_split?
;sequential_1/gru_1/while/gru_cell_1/MatMul_1/ReadVariableOpReadVariableOpFsequential_1_gru_1_while_gru_cell_1_matmul_1_readvariableop_resource_0*
_output_shapes

:<*
dtype0?
,sequential_1/gru_1/while/gru_cell_1/MatMul_1MatMul&sequential_1_gru_1_while_placeholder_2Csequential_1/gru_1/while/gru_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<~
)sequential_1/gru_1/while/gru_cell_1/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ?????
5sequential_1/gru_1/while/gru_cell_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
+sequential_1/gru_1/while/gru_cell_1/split_1SplitV6sequential_1/gru_1/while/gru_cell_1/MatMul_1:product:02sequential_1/gru_1/while/gru_cell_1/Const:output:0>sequential_1/gru_1/while/gru_cell_1/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:?????????:?????????:?????????*
	num_split?
'sequential_1/gru_1/while/gru_cell_1/addAddV22sequential_1/gru_1/while/gru_cell_1/split:output:04sequential_1/gru_1/while/gru_cell_1/split_1:output:0*
T0*'
_output_shapes
:??????????
+sequential_1/gru_1/while/gru_cell_1/SigmoidSigmoid+sequential_1/gru_1/while/gru_cell_1/add:z:0*
T0*'
_output_shapes
:??????????
)sequential_1/gru_1/while/gru_cell_1/add_1AddV22sequential_1/gru_1/while/gru_cell_1/split:output:14sequential_1/gru_1/while/gru_cell_1/split_1:output:1*
T0*'
_output_shapes
:??????????
-sequential_1/gru_1/while/gru_cell_1/Sigmoid_1Sigmoid-sequential_1/gru_1/while/gru_cell_1/add_1:z:0*
T0*'
_output_shapes
:??????????
'sequential_1/gru_1/while/gru_cell_1/mulMul1sequential_1/gru_1/while/gru_cell_1/Sigmoid_1:y:04sequential_1/gru_1/while/gru_cell_1/split_1:output:2*
T0*'
_output_shapes
:??????????
)sequential_1/gru_1/while/gru_cell_1/add_2AddV22sequential_1/gru_1/while/gru_cell_1/split:output:2+sequential_1/gru_1/while/gru_cell_1/mul:z:0*
T0*'
_output_shapes
:??????????
(sequential_1/gru_1/while/gru_cell_1/TanhTanh-sequential_1/gru_1/while/gru_cell_1/add_2:z:0*
T0*'
_output_shapes
:??????????
)sequential_1/gru_1/while/gru_cell_1/mul_1Mul/sequential_1/gru_1/while/gru_cell_1/Sigmoid:y:0&sequential_1_gru_1_while_placeholder_2*
T0*'
_output_shapes
:?????????n
)sequential_1/gru_1/while/gru_cell_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
'sequential_1/gru_1/while/gru_cell_1/subSub2sequential_1/gru_1/while/gru_cell_1/sub/x:output:0/sequential_1/gru_1/while/gru_cell_1/Sigmoid:y:0*
T0*'
_output_shapes
:??????????
)sequential_1/gru_1/while/gru_cell_1/mul_2Mul+sequential_1/gru_1/while/gru_cell_1/sub:z:0,sequential_1/gru_1/while/gru_cell_1/Tanh:y:0*
T0*'
_output_shapes
:??????????
)sequential_1/gru_1/while/gru_cell_1/add_3AddV2-sequential_1/gru_1/while/gru_cell_1/mul_1:z:0-sequential_1/gru_1/while/gru_cell_1/mul_2:z:0*
T0*'
_output_shapes
:??????????
=sequential_1/gru_1/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem&sequential_1_gru_1_while_placeholder_1$sequential_1_gru_1_while_placeholder-sequential_1/gru_1/while/gru_cell_1/add_3:z:0*
_output_shapes
: *
element_dtype0:???`
sequential_1/gru_1/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
sequential_1/gru_1/while/addAddV2$sequential_1_gru_1_while_placeholder'sequential_1/gru_1/while/add/y:output:0*
T0*
_output_shapes
: b
 sequential_1/gru_1/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :?
sequential_1/gru_1/while/add_1AddV2>sequential_1_gru_1_while_sequential_1_gru_1_while_loop_counter)sequential_1/gru_1/while/add_1/y:output:0*
T0*
_output_shapes
: ?
!sequential_1/gru_1/while/IdentityIdentity"sequential_1/gru_1/while/add_1:z:0^sequential_1/gru_1/while/NoOp*
T0*
_output_shapes
: ?
#sequential_1/gru_1/while/Identity_1IdentityDsequential_1_gru_1_while_sequential_1_gru_1_while_maximum_iterations^sequential_1/gru_1/while/NoOp*
T0*
_output_shapes
: ?
#sequential_1/gru_1/while/Identity_2Identity sequential_1/gru_1/while/add:z:0^sequential_1/gru_1/while/NoOp*
T0*
_output_shapes
: ?
#sequential_1/gru_1/while/Identity_3IdentityMsequential_1/gru_1/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^sequential_1/gru_1/while/NoOp*
T0*
_output_shapes
: ?
#sequential_1/gru_1/while/Identity_4Identity-sequential_1/gru_1/while/gru_cell_1/add_3:z:0^sequential_1/gru_1/while/NoOp*
T0*'
_output_shapes
:??????????
sequential_1/gru_1/while/NoOpNoOp:^sequential_1/gru_1/while/gru_cell_1/MatMul/ReadVariableOp<^sequential_1/gru_1/while/gru_cell_1/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "?
Dsequential_1_gru_1_while_gru_cell_1_matmul_1_readvariableop_resourceFsequential_1_gru_1_while_gru_cell_1_matmul_1_readvariableop_resource_0"?
Bsequential_1_gru_1_while_gru_cell_1_matmul_readvariableop_resourceDsequential_1_gru_1_while_gru_cell_1_matmul_readvariableop_resource_0"O
!sequential_1_gru_1_while_identity*sequential_1/gru_1/while/Identity:output:0"S
#sequential_1_gru_1_while_identity_1,sequential_1/gru_1/while/Identity_1:output:0"S
#sequential_1_gru_1_while_identity_2,sequential_1/gru_1/while/Identity_2:output:0"S
#sequential_1_gru_1_while_identity_3,sequential_1/gru_1/while/Identity_3:output:0"S
#sequential_1_gru_1_while_identity_4,sequential_1/gru_1/while/Identity_4:output:0"|
;sequential_1_gru_1_while_sequential_1_gru_1_strided_slice_1=sequential_1_gru_1_while_sequential_1_gru_1_strided_slice_1_0"?
wsequential_1_gru_1_while_tensorarrayv2read_tensorlistgetitem_sequential_1_gru_1_tensorarrayunstack_tensorlistfromtensorysequential_1_gru_1_while_tensorarrayv2read_tensorlistgetitem_sequential_1_gru_1_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#: : : : :?????????: : : : 2v
9sequential_1/gru_1/while/gru_cell_1/MatMul/ReadVariableOp9sequential_1/gru_1/while/gru_cell_1/MatMul/ReadVariableOp2z
;sequential_1/gru_1/while/gru_cell_1/MatMul_1/ReadVariableOp;sequential_1/gru_1/while/gru_cell_1/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
: 
?;
?
gru_1_while_body_10846(
$gru_1_while_gru_1_while_loop_counter.
*gru_1_while_gru_1_while_maximum_iterations
gru_1_while_placeholder
gru_1_while_placeholder_1
gru_1_while_placeholder_2'
#gru_1_while_gru_1_strided_slice_1_0c
_gru_1_while_tensorarrayv2read_tensorlistgetitem_gru_1_tensorarrayunstack_tensorlistfromtensor_0I
7gru_1_while_gru_cell_1_matmul_readvariableop_resource_0:<K
9gru_1_while_gru_cell_1_matmul_1_readvariableop_resource_0:<
gru_1_while_identity
gru_1_while_identity_1
gru_1_while_identity_2
gru_1_while_identity_3
gru_1_while_identity_4%
!gru_1_while_gru_1_strided_slice_1a
]gru_1_while_tensorarrayv2read_tensorlistgetitem_gru_1_tensorarrayunstack_tensorlistfromtensorG
5gru_1_while_gru_cell_1_matmul_readvariableop_resource:<I
7gru_1_while_gru_cell_1_matmul_1_readvariableop_resource:<??,gru_1/while/gru_cell_1/MatMul/ReadVariableOp?.gru_1/while/gru_cell_1/MatMul_1/ReadVariableOp?
=gru_1/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
/gru_1/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem_gru_1_while_tensorarrayv2read_tensorlistgetitem_gru_1_tensorarrayunstack_tensorlistfromtensor_0gru_1_while_placeholderFgru_1/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0?
,gru_1/while/gru_cell_1/MatMul/ReadVariableOpReadVariableOp7gru_1_while_gru_cell_1_matmul_readvariableop_resource_0*
_output_shapes

:<*
dtype0?
gru_1/while/gru_cell_1/MatMulMatMul6gru_1/while/TensorArrayV2Read/TensorListGetItem:item:04gru_1/while/gru_cell_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<q
&gru_1/while/gru_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
gru_1/while/gru_cell_1/splitSplit/gru_1/while/gru_cell_1/split/split_dim:output:0'gru_1/while/gru_cell_1/MatMul:product:0*
T0*M
_output_shapes;
9:?????????:?????????:?????????*
	num_split?
.gru_1/while/gru_cell_1/MatMul_1/ReadVariableOpReadVariableOp9gru_1_while_gru_cell_1_matmul_1_readvariableop_resource_0*
_output_shapes

:<*
dtype0?
gru_1/while/gru_cell_1/MatMul_1MatMulgru_1_while_placeholder_26gru_1/while/gru_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<q
gru_1/while/gru_cell_1/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ????s
(gru_1/while/gru_cell_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
gru_1/while/gru_cell_1/split_1SplitV)gru_1/while/gru_cell_1/MatMul_1:product:0%gru_1/while/gru_cell_1/Const:output:01gru_1/while/gru_cell_1/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:?????????:?????????:?????????*
	num_split?
gru_1/while/gru_cell_1/addAddV2%gru_1/while/gru_cell_1/split:output:0'gru_1/while/gru_cell_1/split_1:output:0*
T0*'
_output_shapes
:?????????{
gru_1/while/gru_cell_1/SigmoidSigmoidgru_1/while/gru_cell_1/add:z:0*
T0*'
_output_shapes
:??????????
gru_1/while/gru_cell_1/add_1AddV2%gru_1/while/gru_cell_1/split:output:1'gru_1/while/gru_cell_1/split_1:output:1*
T0*'
_output_shapes
:?????????
 gru_1/while/gru_cell_1/Sigmoid_1Sigmoid gru_1/while/gru_cell_1/add_1:z:0*
T0*'
_output_shapes
:??????????
gru_1/while/gru_cell_1/mulMul$gru_1/while/gru_cell_1/Sigmoid_1:y:0'gru_1/while/gru_cell_1/split_1:output:2*
T0*'
_output_shapes
:??????????
gru_1/while/gru_cell_1/add_2AddV2%gru_1/while/gru_cell_1/split:output:2gru_1/while/gru_cell_1/mul:z:0*
T0*'
_output_shapes
:?????????w
gru_1/while/gru_cell_1/TanhTanh gru_1/while/gru_cell_1/add_2:z:0*
T0*'
_output_shapes
:??????????
gru_1/while/gru_cell_1/mul_1Mul"gru_1/while/gru_cell_1/Sigmoid:y:0gru_1_while_placeholder_2*
T0*'
_output_shapes
:?????????a
gru_1/while/gru_cell_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
gru_1/while/gru_cell_1/subSub%gru_1/while/gru_cell_1/sub/x:output:0"gru_1/while/gru_cell_1/Sigmoid:y:0*
T0*'
_output_shapes
:??????????
gru_1/while/gru_cell_1/mul_2Mulgru_1/while/gru_cell_1/sub:z:0gru_1/while/gru_cell_1/Tanh:y:0*
T0*'
_output_shapes
:??????????
gru_1/while/gru_cell_1/add_3AddV2 gru_1/while/gru_cell_1/mul_1:z:0 gru_1/while/gru_cell_1/mul_2:z:0*
T0*'
_output_shapes
:??????????
0gru_1/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemgru_1_while_placeholder_1gru_1_while_placeholder gru_1/while/gru_cell_1/add_3:z:0*
_output_shapes
: *
element_dtype0:???S
gru_1/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :n
gru_1/while/addAddV2gru_1_while_placeholdergru_1/while/add/y:output:0*
T0*
_output_shapes
: U
gru_1/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
gru_1/while/add_1AddV2$gru_1_while_gru_1_while_loop_countergru_1/while/add_1/y:output:0*
T0*
_output_shapes
: k
gru_1/while/IdentityIdentitygru_1/while/add_1:z:0^gru_1/while/NoOp*
T0*
_output_shapes
: ?
gru_1/while/Identity_1Identity*gru_1_while_gru_1_while_maximum_iterations^gru_1/while/NoOp*
T0*
_output_shapes
: k
gru_1/while/Identity_2Identitygru_1/while/add:z:0^gru_1/while/NoOp*
T0*
_output_shapes
: ?
gru_1/while/Identity_3Identity@gru_1/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^gru_1/while/NoOp*
T0*
_output_shapes
: ?
gru_1/while/Identity_4Identity gru_1/while/gru_cell_1/add_3:z:0^gru_1/while/NoOp*
T0*'
_output_shapes
:??????????
gru_1/while/NoOpNoOp-^gru_1/while/gru_cell_1/MatMul/ReadVariableOp/^gru_1/while/gru_cell_1/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "H
!gru_1_while_gru_1_strided_slice_1#gru_1_while_gru_1_strided_slice_1_0"t
7gru_1_while_gru_cell_1_matmul_1_readvariableop_resource9gru_1_while_gru_cell_1_matmul_1_readvariableop_resource_0"p
5gru_1_while_gru_cell_1_matmul_readvariableop_resource7gru_1_while_gru_cell_1_matmul_readvariableop_resource_0"5
gru_1_while_identitygru_1/while/Identity:output:0"9
gru_1_while_identity_1gru_1/while/Identity_1:output:0"9
gru_1_while_identity_2gru_1/while/Identity_2:output:0"9
gru_1_while_identity_3gru_1/while/Identity_3:output:0"9
gru_1_while_identity_4gru_1/while/Identity_4:output:0"?
]gru_1_while_tensorarrayv2read_tensorlistgetitem_gru_1_tensorarrayunstack_tensorlistfromtensor_gru_1_while_tensorarrayv2read_tensorlistgetitem_gru_1_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#: : : : :?????????: : : : 2\
,gru_1/while/gru_cell_1/MatMul/ReadVariableOp,gru_1/while/gru_cell_1/MatMul/ReadVariableOp2`
.gru_1/while/gru_cell_1/MatMul_1/ReadVariableOp.gru_1/while/gru_cell_1/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
: 
?G
?
@__inference_gru_1_layer_call_and_return_conditional_losses_11116
inputs_0;
)gru_cell_1_matmul_readvariableop_resource:<=
+gru_cell_1_matmul_1_readvariableop_resource:<
identity?? gru_cell_1/MatMul/ReadVariableOp?"gru_cell_1/MatMul_1/ReadVariableOp?while=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask?
 gru_cell_1/MatMul/ReadVariableOpReadVariableOp)gru_cell_1_matmul_readvariableop_resource*
_output_shapes

:<*
dtype0?
gru_cell_1/MatMulMatMulstrided_slice_2:output:0(gru_cell_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<e
gru_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
gru_cell_1/splitSplit#gru_cell_1/split/split_dim:output:0gru_cell_1/MatMul:product:0*
T0*M
_output_shapes;
9:?????????:?????????:?????????*
	num_split?
"gru_cell_1/MatMul_1/ReadVariableOpReadVariableOp+gru_cell_1_matmul_1_readvariableop_resource*
_output_shapes

:<*
dtype0?
gru_cell_1/MatMul_1MatMulzeros:output:0*gru_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<e
gru_cell_1/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ????g
gru_cell_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
gru_cell_1/split_1SplitVgru_cell_1/MatMul_1:product:0gru_cell_1/Const:output:0%gru_cell_1/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:?????????:?????????:?????????*
	num_split?
gru_cell_1/addAddV2gru_cell_1/split:output:0gru_cell_1/split_1:output:0*
T0*'
_output_shapes
:?????????c
gru_cell_1/SigmoidSigmoidgru_cell_1/add:z:0*
T0*'
_output_shapes
:??????????
gru_cell_1/add_1AddV2gru_cell_1/split:output:1gru_cell_1/split_1:output:1*
T0*'
_output_shapes
:?????????g
gru_cell_1/Sigmoid_1Sigmoidgru_cell_1/add_1:z:0*
T0*'
_output_shapes
:?????????~
gru_cell_1/mulMulgru_cell_1/Sigmoid_1:y:0gru_cell_1/split_1:output:2*
T0*'
_output_shapes
:?????????z
gru_cell_1/add_2AddV2gru_cell_1/split:output:2gru_cell_1/mul:z:0*
T0*'
_output_shapes
:?????????_
gru_cell_1/TanhTanhgru_cell_1/add_2:z:0*
T0*'
_output_shapes
:?????????q
gru_cell_1/mul_1Mulgru_cell_1/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:?????????U
gru_cell_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??z
gru_cell_1/subSubgru_cell_1/sub/x:output:0gru_cell_1/Sigmoid:y:0*
T0*'
_output_shapes
:?????????r
gru_cell_1/mul_2Mulgru_cell_1/sub:z:0gru_cell_1/Tanh:y:0*
T0*'
_output_shapes
:?????????w
gru_cell_1/add_3AddV2gru_cell_1/mul_1:z:0gru_cell_1/mul_2:z:0*
T0*'
_output_shapes
:?????????n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)gru_cell_1_matmul_readvariableop_resource+gru_cell_1_matmul_1_readvariableop_resource*
T
2	*
_lower_using_switch_merge(*
_num_original_outputs	*7
_output_shapes%
#: : : : :?????????: : : : *$
_read_only_resource_inputs
*
_stateful_parallelism( *
bodyR
while_body_11035*
condR
while_cond_11034*6
output_shapes%
#: : : : :?????????: : : : *
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp!^gru_cell_1/MatMul/ReadVariableOp#^gru_cell_1/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????????????: : 2D
 gru_cell_1/MatMul/ReadVariableOp gru_cell_1/MatMul/ReadVariableOp2H
"gru_cell_1/MatMul_1/ReadVariableOp"gru_cell_1/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :??????????????????
"
_user_specified_name
inputs/0
?
?
,__inference_sequential_1_layer_call_fn_10642

inputs
unknown:<
	unknown_0:<
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_10541o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?F
?
@__inference_gru_1_layer_call_and_return_conditional_losses_10280

inputs;
)gru_cell_1_matmul_readvariableop_resource:<=
+gru_cell_1_matmul_1_readvariableop_resource:<
identity?? gru_cell_1/MatMul/ReadVariableOp?"gru_cell_1/MatMul_1/ReadVariableOp?while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:?????????D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask?
 gru_cell_1/MatMul/ReadVariableOpReadVariableOp)gru_cell_1_matmul_readvariableop_resource*
_output_shapes

:<*
dtype0?
gru_cell_1/MatMulMatMulstrided_slice_2:output:0(gru_cell_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<e
gru_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
gru_cell_1/splitSplit#gru_cell_1/split/split_dim:output:0gru_cell_1/MatMul:product:0*
T0*M
_output_shapes;
9:?????????:?????????:?????????*
	num_split?
"gru_cell_1/MatMul_1/ReadVariableOpReadVariableOp+gru_cell_1_matmul_1_readvariableop_resource*
_output_shapes

:<*
dtype0?
gru_cell_1/MatMul_1MatMulzeros:output:0*gru_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<e
gru_cell_1/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ????g
gru_cell_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
gru_cell_1/split_1SplitVgru_cell_1/MatMul_1:product:0gru_cell_1/Const:output:0%gru_cell_1/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:?????????:?????????:?????????*
	num_split?
gru_cell_1/addAddV2gru_cell_1/split:output:0gru_cell_1/split_1:output:0*
T0*'
_output_shapes
:?????????c
gru_cell_1/SigmoidSigmoidgru_cell_1/add:z:0*
T0*'
_output_shapes
:??????????
gru_cell_1/add_1AddV2gru_cell_1/split:output:1gru_cell_1/split_1:output:1*
T0*'
_output_shapes
:?????????g
gru_cell_1/Sigmoid_1Sigmoidgru_cell_1/add_1:z:0*
T0*'
_output_shapes
:?????????~
gru_cell_1/mulMulgru_cell_1/Sigmoid_1:y:0gru_cell_1/split_1:output:2*
T0*'
_output_shapes
:?????????z
gru_cell_1/add_2AddV2gru_cell_1/split:output:2gru_cell_1/mul:z:0*
T0*'
_output_shapes
:?????????_
gru_cell_1/TanhTanhgru_cell_1/add_2:z:0*
T0*'
_output_shapes
:?????????q
gru_cell_1/mul_1Mulgru_cell_1/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:?????????U
gru_cell_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??z
gru_cell_1/subSubgru_cell_1/sub/x:output:0gru_cell_1/Sigmoid:y:0*
T0*'
_output_shapes
:?????????r
gru_cell_1/mul_2Mulgru_cell_1/sub:z:0gru_cell_1/Tanh:y:0*
T0*'
_output_shapes
:?????????w
gru_cell_1/add_3AddV2gru_cell_1/mul_1:z:0gru_cell_1/mul_2:z:0*
T0*'
_output_shapes
:?????????n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)gru_cell_1_matmul_readvariableop_resource+gru_cell_1_matmul_1_readvariableop_resource*
T
2	*
_lower_using_switch_merge(*
_num_original_outputs	*7
_output_shapes%
#: : : : :?????????: : : : *$
_read_only_resource_inputs
*
_stateful_parallelism( *
bodyR
while_body_10199*
condR
while_cond_10198*6
output_shapes%
#: : : : :?????????: : : : *
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp!^gru_cell_1/MatMul/ReadVariableOp#^gru_cell_1/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : 2D
 gru_cell_1/MatMul/ReadVariableOp gru_cell_1/MatMul/ReadVariableOp2H
"gru_cell_1/MatMul_1/ReadVariableOp"gru_cell_1/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?4
?
while_body_11452
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0C
1while_gru_cell_1_matmul_readvariableop_resource_0:<E
3while_gru_cell_1_matmul_1_readvariableop_resource_0:<
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorA
/while_gru_cell_1_matmul_readvariableop_resource:<C
1while_gru_cell_1_matmul_1_readvariableop_resource:<??&while/gru_cell_1/MatMul/ReadVariableOp?(while/gru_cell_1/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0?
&while/gru_cell_1/MatMul/ReadVariableOpReadVariableOp1while_gru_cell_1_matmul_readvariableop_resource_0*
_output_shapes

:<*
dtype0?
while/gru_cell_1/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/gru_cell_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<k
 while/gru_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
while/gru_cell_1/splitSplit)while/gru_cell_1/split/split_dim:output:0!while/gru_cell_1/MatMul:product:0*
T0*M
_output_shapes;
9:?????????:?????????:?????????*
	num_split?
(while/gru_cell_1/MatMul_1/ReadVariableOpReadVariableOp3while_gru_cell_1_matmul_1_readvariableop_resource_0*
_output_shapes

:<*
dtype0?
while/gru_cell_1/MatMul_1MatMulwhile_placeholder_20while/gru_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<k
while/gru_cell_1/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ????m
"while/gru_cell_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
while/gru_cell_1/split_1SplitV#while/gru_cell_1/MatMul_1:product:0while/gru_cell_1/Const:output:0+while/gru_cell_1/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:?????????:?????????:?????????*
	num_split?
while/gru_cell_1/addAddV2while/gru_cell_1/split:output:0!while/gru_cell_1/split_1:output:0*
T0*'
_output_shapes
:?????????o
while/gru_cell_1/SigmoidSigmoidwhile/gru_cell_1/add:z:0*
T0*'
_output_shapes
:??????????
while/gru_cell_1/add_1AddV2while/gru_cell_1/split:output:1!while/gru_cell_1/split_1:output:1*
T0*'
_output_shapes
:?????????s
while/gru_cell_1/Sigmoid_1Sigmoidwhile/gru_cell_1/add_1:z:0*
T0*'
_output_shapes
:??????????
while/gru_cell_1/mulMulwhile/gru_cell_1/Sigmoid_1:y:0!while/gru_cell_1/split_1:output:2*
T0*'
_output_shapes
:??????????
while/gru_cell_1/add_2AddV2while/gru_cell_1/split:output:2while/gru_cell_1/mul:z:0*
T0*'
_output_shapes
:?????????k
while/gru_cell_1/TanhTanhwhile/gru_cell_1/add_2:z:0*
T0*'
_output_shapes
:??????????
while/gru_cell_1/mul_1Mulwhile/gru_cell_1/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:?????????[
while/gru_cell_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
while/gru_cell_1/subSubwhile/gru_cell_1/sub/x:output:0while/gru_cell_1/Sigmoid:y:0*
T0*'
_output_shapes
:??????????
while/gru_cell_1/mul_2Mulwhile/gru_cell_1/sub:z:0while/gru_cell_1/Tanh:y:0*
T0*'
_output_shapes
:??????????
while/gru_cell_1/add_3AddV2while/gru_cell_1/mul_1:z:0while/gru_cell_1/mul_2:z:0*
T0*'
_output_shapes
:??????????
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_1/add_3:z:0*
_output_shapes
: *
element_dtype0:???M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: ?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: w
while/Identity_4Identitywhile/gru_cell_1/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:??????????

while/NoOpNoOp'^while/gru_cell_1/MatMul/ReadVariableOp)^while/gru_cell_1/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "h
1while_gru_cell_1_matmul_1_readvariableop_resource3while_gru_cell_1_matmul_1_readvariableop_resource_0"d
/while_gru_cell_1_matmul_readvariableop_resource1while_gru_cell_1_matmul_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#: : : : :?????????: : : : 2P
&while/gru_cell_1/MatMul/ReadVariableOp&while/gru_cell_1/MatMul/ReadVariableOp2T
(while/gru_cell_1/MatMul_1/ReadVariableOp(while/gru_cell_1/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
: 
?

?
*__inference_gru_cell_1_layer_call_fn_11603

inputs
states_0
unknown:<
	unknown_0:<
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_gru_cell_1_layer_call_and_return_conditional_losses_10020o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*=
_input_shapes,
*:?????????:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:QM
'
_output_shapes
:?????????
"
_user_specified_name
states/0"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
G
gru_1_input8
serving_default_gru_1_input:0?????????;
dense_10
StatefulPartitionedCall:0?????????tensorflow/serving/predict:?m
?
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api
	
signatures
L__call__
*M&call_and_return_all_conditional_losses
N_default_save_signature"
_tf_keras_sequential
?

cell

state_spec
	variables
trainable_variables
regularization_losses
	keras_api
O__call__
*P&call_and_return_all_conditional_losses"
_tf_keras_rnn_layer
?
	variables
trainable_variables
regularization_losses
	keras_api
Q__call__
*R&call_and_return_all_conditional_losses"
_tf_keras_layer
?

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
S__call__
*T&call_and_return_all_conditional_losses"
_tf_keras_layer
?
iter

beta_1

beta_2
	decay
learning_ratemDmEmF mGvHvIvJ vK"
	optimizer
<
0
 1
2
3"
trackable_list_wrapper
<
0
 1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
?
!non_trainable_variables

"layers
#metrics
$layer_regularization_losses
%layer_metrics
	variables
trainable_variables
regularization_losses
L__call__
N_default_save_signature
*M&call_and_return_all_conditional_losses
&M"call_and_return_conditional_losses"
_generic_user_object
,
Userving_default"
signature_map
?

kernel
 recurrent_kernel
&	variables
'trainable_variables
(regularization_losses
)	keras_api
V__call__
*W&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_list_wrapper
.
0
 1"
trackable_list_wrapper
.
0
 1"
trackable_list_wrapper
 "
trackable_list_wrapper
?

*states
+non_trainable_variables

,layers
-metrics
.layer_regularization_losses
/layer_metrics
	variables
trainable_variables
regularization_losses
O__call__
*P&call_and_return_all_conditional_losses
&P"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
0non_trainable_variables

1layers
2metrics
3layer_regularization_losses
4layer_metrics
	variables
trainable_variables
regularization_losses
Q__call__
*R&call_and_return_all_conditional_losses
&R"call_and_return_conditional_losses"
_generic_user_object
 :2dense_1/kernel
:2dense_1/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
5non_trainable_variables

6layers
7metrics
8layer_regularization_losses
9layer_metrics
	variables
trainable_variables
regularization_losses
S__call__
*T&call_and_return_all_conditional_losses
&T"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
):'<2gru_1/gru_cell_1/kernel
3:1<2!gru_1/gru_cell_1/recurrent_kernel
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
'
:0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
0
 1"
trackable_list_wrapper
.
0
 1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
;non_trainable_variables

<layers
=metrics
>layer_regularization_losses
?layer_metrics
&	variables
'trainable_variables
(regularization_losses
V__call__
*W&call_and_return_all_conditional_losses
&W"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'

0"
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
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
N
	@total
	Acount
B	variables
C	keras_api"
_tf_keras_metric
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
:  (2total
:  (2count
.
@0
A1"
trackable_list_wrapper
-
B	variables"
_generic_user_object
%:#2Adam/dense_1/kernel/m
:2Adam/dense_1/bias/m
.:,<2Adam/gru_1/gru_cell_1/kernel/m
8:6<2(Adam/gru_1/gru_cell_1/recurrent_kernel/m
%:#2Adam/dense_1/kernel/v
:2Adam/dense_1/bias/v
.:,<2Adam/gru_1/gru_cell_1/kernel/v
8:6<2(Adam/gru_1/gru_cell_1/recurrent_kernel/v
?2?
,__inference_sequential_1_layer_call_fn_10321
,__inference_sequential_1_layer_call_fn_10629
,__inference_sequential_1_layer_call_fn_10642
,__inference_sequential_1_layer_call_fn_10565?
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
G__inference_sequential_1_layer_call_and_return_conditional_losses_10788
G__inference_sequential_1_layer_call_and_return_conditional_losses_10941
G__inference_sequential_1_layer_call_and_return_conditional_losses_10580
G__inference_sequential_1_layer_call_and_return_conditional_losses_10595?
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
?B?
__inference__wrapped_model_9829gru_1_input"?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
%__inference_gru_1_layer_call_fn_10950
%__inference_gru_1_layer_call_fn_10959
%__inference_gru_1_layer_call_fn_10968
%__inference_gru_1_layer_call_fn_10977?
???
FullArgSpecB
args:?7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults?

 
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
@__inference_gru_1_layer_call_and_return_conditional_losses_11116
@__inference_gru_1_layer_call_and_return_conditional_losses_11255
@__inference_gru_1_layer_call_and_return_conditional_losses_11394
@__inference_gru_1_layer_call_and_return_conditional_losses_11533?
???
FullArgSpecB
args:?7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults?

 
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
)__inference_dropout_1_layer_call_fn_11538
)__inference_dropout_1_layer_call_fn_11543?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
D__inference_dropout_1_layer_call_and_return_conditional_losses_11548
D__inference_dropout_1_layer_call_and_return_conditional_losses_11560?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
'__inference_dense_1_layer_call_fn_11569?
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
B__inference_dense_1_layer_call_and_return_conditional_losses_11579?
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
?B?
#__inference_signature_wrapper_10616gru_1_input"?
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
?2?
*__inference_gru_cell_1_layer_call_fn_11591
*__inference_gru_cell_1_layer_call_fn_11603?
???
FullArgSpec3
args+?(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
E__inference_gru_cell_1_layer_call_and_return_conditional_losses_11636
E__inference_gru_cell_1_layer_call_and_return_conditional_losses_11669?
???
FullArgSpec3
args+?(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 ?
__inference__wrapped_model_9829s 8?5
.?+
)?&
gru_1_input?????????
? "1?.
,
dense_1!?
dense_1??????????
B__inference_dense_1_layer_call_and_return_conditional_losses_11579\/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? z
'__inference_dense_1_layer_call_fn_11569O/?,
%?"
 ?
inputs?????????
? "???????????
D__inference_dropout_1_layer_call_and_return_conditional_losses_11548\3?0
)?&
 ?
inputs?????????
p 
? "%?"
?
0?????????
? ?
D__inference_dropout_1_layer_call_and_return_conditional_losses_11560\3?0
)?&
 ?
inputs?????????
p
? "%?"
?
0?????????
? |
)__inference_dropout_1_layer_call_fn_11538O3?0
)?&
 ?
inputs?????????
p 
? "??????????|
)__inference_dropout_1_layer_call_fn_11543O3?0
)?&
 ?
inputs?????????
p
? "???????????
@__inference_gru_1_layer_call_and_return_conditional_losses_11116| O?L
E?B
4?1
/?,
inputs/0??????????????????

 
p 

 
? "%?"
?
0?????????
? ?
@__inference_gru_1_layer_call_and_return_conditional_losses_11255| O?L
E?B
4?1
/?,
inputs/0??????????????????

 
p

 
? "%?"
?
0?????????
? ?
@__inference_gru_1_layer_call_and_return_conditional_losses_11394l ??<
5?2
$?!
inputs?????????

 
p 

 
? "%?"
?
0?????????
? ?
@__inference_gru_1_layer_call_and_return_conditional_losses_11533l ??<
5?2
$?!
inputs?????????

 
p

 
? "%?"
?
0?????????
? ?
%__inference_gru_1_layer_call_fn_10950o O?L
E?B
4?1
/?,
inputs/0??????????????????

 
p 

 
? "???????????
%__inference_gru_1_layer_call_fn_10959o O?L
E?B
4?1
/?,
inputs/0??????????????????

 
p

 
? "???????????
%__inference_gru_1_layer_call_fn_10968_ ??<
5?2
$?!
inputs?????????

 
p 

 
? "???????????
%__inference_gru_1_layer_call_fn_10977_ ??<
5?2
$?!
inputs?????????

 
p

 
? "???????????
E__inference_gru_cell_1_layer_call_and_return_conditional_losses_11636? \?Y
R?O
 ?
inputs?????????
'?$
"?
states/0?????????
p 
? "R?O
H?E
?
0/0?????????
$?!
?
0/1/0?????????
? ?
E__inference_gru_cell_1_layer_call_and_return_conditional_losses_11669? \?Y
R?O
 ?
inputs?????????
'?$
"?
states/0?????????
p
? "R?O
H?E
?
0/0?????????
$?!
?
0/1/0?????????
? ?
*__inference_gru_cell_1_layer_call_fn_11591? \?Y
R?O
 ?
inputs?????????
'?$
"?
states/0?????????
p 
? "D?A
?
0?????????
"?
?
1/0??????????
*__inference_gru_cell_1_layer_call_fn_11603? \?Y
R?O
 ?
inputs?????????
'?$
"?
states/0?????????
p
? "D?A
?
0?????????
"?
?
1/0??????????
G__inference_sequential_1_layer_call_and_return_conditional_losses_10580o @?=
6?3
)?&
gru_1_input?????????
p 

 
? "%?"
?
0?????????
? ?
G__inference_sequential_1_layer_call_and_return_conditional_losses_10595o @?=
6?3
)?&
gru_1_input?????????
p

 
? "%?"
?
0?????????
? ?
G__inference_sequential_1_layer_call_and_return_conditional_losses_10788j ;?8
1?.
$?!
inputs?????????
p 

 
? "%?"
?
0?????????
? ?
G__inference_sequential_1_layer_call_and_return_conditional_losses_10941j ;?8
1?.
$?!
inputs?????????
p

 
? "%?"
?
0?????????
? ?
,__inference_sequential_1_layer_call_fn_10321b @?=
6?3
)?&
gru_1_input?????????
p 

 
? "???????????
,__inference_sequential_1_layer_call_fn_10565b @?=
6?3
)?&
gru_1_input?????????
p

 
? "???????????
,__inference_sequential_1_layer_call_fn_10629] ;?8
1?.
$?!
inputs?????????
p 

 
? "???????????
,__inference_sequential_1_layer_call_fn_10642] ;?8
1?.
$?!
inputs?????????
p

 
? "???????????
#__inference_signature_wrapper_10616? G?D
? 
=?:
8
gru_1_input)?&
gru_1_input?????????"1?.
,
dense_1!?
dense_1?????????