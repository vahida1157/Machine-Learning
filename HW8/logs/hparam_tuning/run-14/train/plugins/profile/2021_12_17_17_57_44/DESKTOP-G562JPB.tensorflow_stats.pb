"?D
BHostIDLE"IDLE13333 G$AA3333 G$Aa?hYQ+???i?hYQ+????Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1?????b?@9?????b?@A?????b?@I?????b?@a?Z ?u?i?i4????????Unknown?
sHostDestroyResourceOp"DestroyResourceOp(1     ?B@9%I?$I???A     ?B@I%I?$I???a?l%_{?i?r?d????Unknown?
?HostResourceApplyAdam""Adam/Adam/update/ResourceApplyAdam(1fffff?>@9fffff?>@Afffff?>@Ifffff?>@a?T??K?i?1???????Unknown
dHostDataset"Iterator::Model(1     ?;@9     ?;@A??????6@I??????6@a6zN???i?iʓ????Unknown
iHostWriteSummary"WriteSummary(1?????L4@9?????L4@A?????L4@I?????L4@a??,?I??>iT?]jM????Unknown?
?HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1      2@9      2@A333333-@I333333-@a?o&???>i??#T{????Unknown
?HostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate(1     ?2@9     ?2@A??????,@I??????,@a? \? |?>iY?eL?????Unknown
u	Host_FusedMatMul"sequential_15/dense_30/Relu(1??????'@9??????'@A??????'@I??????'@a?????>ij???????Unknown
?
HostResourceApplyAdam"$Adam/Adam/update_3/ResourceApplyAdam(1??????&@9??????&@A??????&@I??????&@a??s????>iQR?A?????Unknown
?HostResourceApplyAdam"$Adam/Adam/update_1/ResourceApplyAdam(1ffffff"@9ffffff"@Affffff"@Iffffff"@a C??|??>i&0????Unknown
^HostGatherV2"GatherV2(1ffffff!@9ffffff!@Affffff!@Iffffff!@aQ&???[?>i?)????Unknown
HostMatMul"+gradient_tape/sequential_15/dense_30/MatMul(1??????@9??????@A??????@I??????@a?+P$???>iS:?cB????Unknown
?HostResourceApplyAdam"$Adam/Adam/update_2/ResourceApplyAdam(1333333@9333333@A333333@I333333@aI??i??>irh?Z????Unknown
HostMatMul"+gradient_tape/sequential_15/dense_31/MatMul(1ffffff@9ffffff@Affffff@Iffffff@a?0?J?S?>i5NH?q????Unknown
xHostDataset"#Iterator::Model::ParallelMapV2::Zip(133333?F@933333?F@A333333@I333333@a?R.9\b?>ic????????Unknown
[HostAddV2"Adam/add(1??????@9??????@A??????@I??????@a??oЍ?>iӋt/?????Unknown
eHost
LogicalAnd"
LogicalAnd(1??????@9??????@A??????@I??????@a6zN???>i??A?????Unknown?
?HostDynamicStitch".gradient_tape/mean_squared_error/DynamicStitch(1ffffff@9ffffff@Affffff@Iffffff@a0??*?	?>i?&?????Unknown
sHostDataset"Iterator::Model::ParallelMapV2(1??????@9??????@A??????@I??????@a??8|??>i?????????Unknown
gHostStridedSlice"strided_slice(1ffffff@9ffffff@Affffff@Iffffff@a C??|??>i?%e?????Unknown
xHost_FusedMatMul"sequential_15/dense_31/BiasAdd(1333333@9333333@A333333@I333333@a????t?>iw???????Unknown
?HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1??????@9??????@A??????@I??????@aOHH?qj?>iCm??????Unknown
lHostIteratorGetNext"IteratorGetNext(1ffffff@9ffffff@Affffff@Iffffff@a?	?5o??>i6? ????Unknown
?HostTile"5gradient_tape/mean_squared_error/weighted_loss/Tile_1(1333333@9333333@A333333@I333333@aI??i??>i??dH????Unknown
?HostBiasAddGrad"8gradient_tape/sequential_15/dense_30/BiasAdd/BiasAddGrad(1333333@9333333@A333333@I333333@aI??i??>iT??????Unknown
`HostGatherV2"
GatherV2_1(1??????@9??????@A??????@I??????@a?X\dE?>i???.$????Unknown
?HostReluGrad"-gradient_tape/sequential_15/dense_30/ReluGrad(1??????@9??????@A??????@I??????@a?X\dE?>i?-~?/????Unknown
VHostSum"Sum_2(1??????@9??????@A??????@I??????@aD???a??>i? ?#;????Unknown
?HostBiasAddGrad"8gradient_tape/sequential_15/dense_31/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a????^?>iox^%F????Unknown
?HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1333333@9333333??A333333@I333333??a?R.9\b?>i???P????Unknown
p HostSquaredDifference"SquaredDifference(1333333@9333333@A333333@I333333@a?R.9\b?>i????[????Unknown
v!HostAssignAddVariableOp"AssignAddVariableOp_2(1333333@9333333@A333333@I333333@a?>?N=?>i<b?d????Unknown
?"HostReadVariableOp"-sequential_15/dense_30/BiasAdd/ReadVariableOp(1333333@9333333@A333333@I333333@a?>?N=?>i?Z	?m????Unknown
V#HostMean"Mean(1??????@9??????@A??????@I??????@a3?w<I??>i??Bv????Unknown
[$HostPow"
Adam/Pow_1(1??????@9??????@A??????@I??????@a?]?FZ?>i!<?o~????Unknown
?%HostMatMul"-gradient_tape/sequential_15/dense_31/MatMul_1(1      @9      @A      @I      @a?=b??r?>i?#sL?????Unknown
?&HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1??????4@9??????4@A?????? @I?????? @aOHH?qj?>i???????Unknown
|'HostDivNoNan"&mean_squared_error/weighted_loss/value(1ffffff??9ffffff??Affffff??Iffffff??a?M?g??>i?c???????Unknown
](HostCast"Adam/Cast_1(1????????9????????A????????I????????aD???a??>i8????????Unknown
})HostMaximum"(gradient_tape/mean_squared_error/Maximum(1????????9????????A????????I????????aD???a??>i?6?2?????Unknown
t*HostAssignAddVariableOp"AssignAddVariableOp(1ffffff??9ffffff??Affffff??Iffffff??a????K??>il3홢????Unknown
V+HostCast"Cast(1ffffff??9ffffff??Affffff??Iffffff??a????K??>i#0 ?????Unknown
v,HostAssignAddVariableOp"AssignAddVariableOp_4(1????????9????????A????????I????????a?]?FZ?>i?ё?????Unknown
u-HostSub"$gradient_tape/mean_squared_error/sub(1????????9????????A????????I????????a?]?FZ?>i-s#.?????Unknown
}.HostRealDiv"(gradient_tape/mean_squared_error/truediv(1????????9????????A????????I????????a?]?FZ?>i??D?????Unknown
Y/HostPow"Adam/Pow(1333333??9333333??A333333??I333333??aZ??2?0?>i[?
?????Unknown
T0HostMul"Mul(1333333??9333333??A333333??I333333??aZ??2?0?>iX??к????Unknown
b1HostDivNoNan"div_no_nan_1(1333333??9333333??A333333??I333333??aZ??2?0?>i??喾????Unknown
2HostFloorDiv")gradient_tape/mean_squared_error/floordiv(1333333??9333333??A333333??I333333??aZ??2?0?>i?-?\?????Unknown
?3HostReadVariableOp",sequential_15/dense_30/MatMul/ReadVariableOp(1333333??9333333??A333333??I333333??aZ??2?0?>iQt#?????Unknown
t4HostReadVariableOp"Adam/Cast/ReadVariableOp(1????????9????????A????????I????????a??Yw??>is_???????Unknown
o5HostReadVariableOp"Adam/ReadVariableOp(1????????9????????A????????I????????a??Yw??>i?J$?????Unknown
?6HostBroadcastTo",gradient_tape/mean_squared_error/BroadcastTo(1????????9????????A????????I????????a??Yw??>i?5???????Unknown
u7HostMul"$gradient_tape/mean_squared_error/Mul(1????????9????????A????????I????????a??Yw??>i? B??????Unknown
u8HostSum"$gradient_tape/mean_squared_error/Sum(1????????9????????A????????I????????a??Yw??>i??n?????Unknown
v9HostAssignAddVariableOp"AssignAddVariableOp_3(1      ??9      ??A      ??I      ??a?ʁl(?>i??ޓ?????Unknown
w:HostReadVariableOp"div_no_nan_1/ReadVariableOp(1      ??9      ??A      ??I      ??a?ʁl(?>i?+???????Unknown
v;HostReadVariableOp"Adam/Cast_2/ReadVariableOp(1????????9????????A????????I????????aD???a??>i?`x??????Unknown
v<HostReadVariableOp"Adam/Cast_3/ReadVariableOp(1????????9????????A????????I????????aD???a??>iY?b?????Unknown
X=HostCast"Cast_1(1????????9????????A????????I????????aD???a??>iʐ6?????Unknown
~>HostAssignAddVariableOp"Adam/Adam/AssignAddVariableOp(1????????9????????A????????I????????a??g?V ?>i?????????Unknown
??HostDivNoNan"?gradient_tape/mean_squared_error/weighted_loss/value/div_no_nan(1????????9????????A????????I????????a??g?V ?>i2}?>?????Unknown
?@HostReadVariableOp"-sequential_15/dense_31/BiasAdd/ReadVariableOp(1????????9????????A????????I????????a??g?V ?>i?V???????Unknown
vAHostAssignAddVariableOp"AssignAddVariableOp_1(1ffffff??9ffffff??Affffff??Iffffff??a????K??>i?:??????Unknown
aBHostIdentity"Identity(1ffffff??9ffffff??Affffff??Iffffff??a????K??>iuS?)?????Unknown?
uCHostReadVariableOp"div_no_nan/ReadVariableOp(1ffffff??9ffffff??Affffff??Iffffff??a????K??>i??M]?????Unknown
yDHostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1ffffff??9ffffff??Affffff??Iffffff??a????K??>i+Pא?????Unknown
wEHostCast"%gradient_tape/mean_squared_error/Cast(1ffffff??9ffffff??Affffff??Iffffff??a????K??>i??`??????Unknown
?FHostReadVariableOp",sequential_15/dense_31/MatMul/ReadVariableOp(1ffffff??9ffffff??Affffff??Iffffff??a????K??>i?L???????Unknown
`GHostDivNoNan"
div_no_nan(1333333??9333333??A333333??I333333??aZ??2?0?>ip???????Unknown
wHHostReadVariableOp"div_no_nan/ReadVariableOp_1(1      ??9      ??A      ??I      ??a?ʁl(?>i8ym?????Unknown
wIHostMul"&gradient_tape/mean_squared_error/mul_1(1      ??9      ??A      ??I      ??a?ʁl(?>i?????????Unknown*?C
uHostFlushSummaryWriter"FlushSummaryWriter(1?????b?@9?????b?@A?????b?@I?????b?@a??F????i??F?????Unknown?
sHostDestroyResourceOp"DestroyResourceOp(1     ?B@9%I?$I???A     ?B@I%I?$I???asv1?1???i???!???Unknown?
?HostResourceApplyAdam""Adam/Adam/update/ResourceApplyAdam(1fffff?>@9fffff?>@Afffff?>@Ifffff?>@a9|r?C??i??	?͉???Unknown
dHostDataset"Iterator::Model(1     ?;@9     ?;@A??????6@I??????6@a:?6?|:??iAbJķ????Unknown
iHostWriteSummary"WriteSummary(1?????L4@9?????L4@A?????L4@I?????L4@aC?f????iv?K?2???Unknown?
?HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1      2@9      2@A333333-@I333333-@aiǱ?<?x?i`?3sL???Unknown
?HostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate(1     ?2@9     ?2@A??????,@I??????,@aM<洲x?i~,'??|???Unknown
uHost_FusedMatMul"sequential_15/dense_30/Relu(1??????'@9??????'@A??????'@I??????'@ai㉖bt?iE@T^դ???Unknown
?	HostResourceApplyAdam"$Adam/Adam/update_3/ResourceApplyAdam(1??????&@9??????&@A??????&@I??????&@a0???Ns?i?%r??????Unknown
?
HostResourceApplyAdam"$Adam/Adam/update_1/ResourceApplyAdam(1ffffff"@9ffffff"@Affffff"@Iffffff"@a????	o?i??Z?????Unknown
^HostGatherV2"GatherV2(1ffffff!@9ffffff!@Affffff!@Iffffff!@ar??DYm?ib7QV???Unknown
HostMatMul"+gradient_tape/sequential_15/dense_30/MatMul(1??????@9??????@A??????@I??????@a????d?j?iV????!???Unknown
?HostResourceApplyAdam"$Adam/Adam/update_2/ResourceApplyAdam(1333333@9333333@A333333@I333333@a?AX?Pj?i??ľL<???Unknown
HostMatMul"+gradient_tape/sequential_15/dense_31/MatMul(1ffffff@9ffffff@Affffff@Iffffff@aCc????g?i??wC@T???Unknown
xHostDataset"#Iterator::Model::ParallelMapV2::Zip(133333?F@933333?F@A333333@I333333@a
M?p?f?iH?#?0k???Unknown
[HostAddV2"Adam/add(1??????@9??????@A??????@I??????@a`
F?4?c?iR3?????Unknown
eHost
LogicalAnd"
LogicalAnd(1??????@9??????@A??????@I??????@a:?6?|:c?i?iIeR????Unknown?
?HostDynamicStitch".gradient_tape/mean_squared_error/DynamicStitch(1ffffff@9ffffff@Affffff@Iffffff@a?y?T4a?irrʹ?????Unknown
sHostDataset"Iterator::Model::ParallelMapV2(1??????@9??????@A??????@I??????@a?b??ȵ_?i#\A?a????Unknown
gHostStridedSlice"strided_slice(1ffffff@9ffffff@Affffff@Iffffff@a????	_?i"??&?????Unknown
xHost_FusedMatMul"sequential_15/dense_31/BiasAdd(1333333@9333333@A333333@I333333@a_ҕ??]?i?"?g????Unknown
?HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1??????@9??????@A??????@I??????@a:n??0V\?iB̌??????Unknown
lHostIteratorGetNext"IteratorGetNext(1ffffff@9ffffff@Affffff@Iffffff@a
w?x?[?iǇ?og????Unknown
?HostTile"5gradient_tape/mean_squared_error/weighted_loss/Tile_1(1333333@9333333@A333333@I333333@a?AX?PZ?i?3Wt?????Unknown
?HostBiasAddGrad"8gradient_tape/sequential_15/dense_30/BiasAdd/BiasAddGrad(1333333@9333333@A333333@I333333@a?AX?PZ?i	??x????Unknown
`HostGatherV2"
GatherV2_1(1??????@9??????@A??????@I??????@a|y9???X?i?|?2???Unknown
?HostReluGrad"-gradient_tape/sequential_15/dense_30/ReluGrad(1??????@9??????@A??????@I??????@a|y9???X?i?u? ???Unknown
VHostSum"Sum_2(1??????@9??????@A??????@I??????@aV*??IX?i?.??,???Unknown
?HostBiasAddGrad"8gradient_tape/sequential_15/dense_31/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a0??(?W?i??(??8???Unknown
?HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1333333@9333333??A333333@I333333??a
M?p?V?i??~?D???Unknown
pHostSquaredDifference"SquaredDifference(1333333@9333333@A333333@I333333@a
M?p?V?i5???O???Unknown
v HostAssignAddVariableOp"AssignAddVariableOp_2(1333333@9333333@A333333@I333333@aLX??ؐS?ia&sZY???Unknown
?!HostReadVariableOp"-sequential_15/dense_30/BiasAdd/ReadVariableOp(1333333@9333333@A333333@I333333@aLX??ؐS?i??g?"c???Unknown
V"HostMean"Mean(1??????@9??????@A??????@I??????@a???h7R?iUի?>l???Unknown
[#HostPow"
Adam/Pow_1(1??????@9??????@A??????@I??????@a?+????Q?ik???u???Unknown
?$HostMatMul"-gradient_tape/sequential_15/dense_31/MatMul_1(1      @9      @A      @I      @a?ǀ~??P?i??,?r}???Unknown
?%HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1??????4@9??????4@A?????? @I?????? @a:n??0VL?ik?at?????Unknown
|&HostDivNoNan"&mean_squared_error/weighted_loss/value(1ffffff??9ffffff??Affffff??Iffffff??a??H?P?I?i??H?????Unknown
]'HostCast"Adam/Cast_1(1????????9????????A????????I????????aV*??IH?i'???????Unknown
}(HostMaximum"(gradient_tape/mean_squared_error/Maximum(1????????9????????A????????I????????aV*??IH?i?&?8????Unknown
t)HostAssignAddVariableOp"AssignAddVariableOp(1ffffff??9ffffff??Affffff??Iffffff??a&??? ?B?ii?Aϛ???Unknown
V*HostCast"Cast(1ffffff??9ffffff??Affffff??Iffffff??a&??? ?B?i&?3I?????Unknown
v+HostAssignAddVariableOp"AssignAddVariableOp_4(1????????9????????A????????I????????a?+????A?i1?T??????Unknown
u,HostSub"$gradient_tape/mean_squared_error/sub(1????????9????????A????????I????????a?+????A?i<?u?M????Unknown
}-HostRealDiv"(gradient_tape/mean_squared_error/truediv(1????????9????????A????????I????????a?+????A?iG??M?????Unknown
Y.HostPow"Adam/Pow(1333333??9333333??A333333??I333333??a?cqy@1@?i????????Unknown
T/HostMul"Mul(1333333??9333333??A333333??I333333??a?cqy@1@?i?b??ȵ???Unknown
b0HostDivNoNan"div_no_nan_1(1333333??9333333??A333333??I333333??a?cqy@1@?iR??=չ???Unknown
1HostFloorDiv")gradient_tape/mean_squared_error/floordiv(1333333??9333333??A333333??I333333??a?cqy@1@?i???????Unknown
?2HostReadVariableOp",sequential_15/dense_30/MatMul/ReadVariableOp(1333333??9333333??A333333??I333333??a?cqy@1@?ix.??????Unknown
t3HostReadVariableOp"Adam/Cast/ReadVariableOp(1????????9????????A????????I????????a?6?ޠ?=?i?LJң????Unknown
o4HostReadVariableOp"Adam/ReadVariableOp(1????????9????????A????????I????????a?6?ޠ?=?iR!f?Y????Unknown
?5HostBroadcastTo",gradient_tape/mean_squared_error/BroadcastTo(1????????9????????A????????I????????a?6?ޠ?=?i????????Unknown
u6HostMul"$gradient_tape/mean_squared_error/Mul(1????????9????????A????????I????????a?6?ޠ?=?i?ʝ??????Unknown
u7HostSum"$gradient_tape/mean_squared_error/Sum(1????????9????????A????????I????????a?6?ޠ?=?iG???{????Unknown
v8HostAssignAddVariableOp"AssignAddVariableOp_3(1      ??9      ??A      ??I      ??a??g???:?i<??:?????Unknown
w9HostReadVariableOp"div_no_nan_1/ReadVariableOp(1      ??9      ??A      ??I      ??a??g???:?i19??:????Unknown
v:HostReadVariableOp"Adam/Cast_2/ReadVariableOp(1????????9????????A????????I????????aV*??I8?it?D????Unknown
v;HostReadVariableOp"Adam/Cast_3/ReadVariableOp(1????????9????????A????????I????????aV*??I8?i??KM????Unknown
X<HostCast"Cast_1(1????????9????????A????????I????????aV*??I8?i??0?V????Unknown
~=HostAssignAddVariableOp"Adam/Adam/AssignAddVariableOp(1????????9????????A????????I????????a???? ?5?i??Dg	????Unknown
?>HostDivNoNan"?gradient_tape/mean_squared_error/weighted_loss/value/div_no_nan(1????????9????????A????????I????????a???? ?5?iYG?????Unknown
??HostReadVariableOp"-sequential_15/dense_31/BiasAdd/ReadVariableOp(1????????9????????A????????I????????a???? ?5?i?Am'o????Unknown
v@HostAssignAddVariableOp"AssignAddVariableOp_1(1ffffff??9ffffff??Affffff??Iffffff??a&??? ?2?i??~??????Unknown
aAHostIdentity"Identity(1ffffff??9ffffff??Affffff??Iffffff??a&??? ?2?ik??/(????Unknown?
uBHostReadVariableOp"div_no_nan/ReadVariableOp(1ffffff??9ffffff??Affffff??Iffffff??a&??? ?2?iJc???????Unknown
yCHostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1ffffff??9ffffff??Affffff??Iffffff??a&??? ?2?i)?7?????Unknown
wDHostCast"%gradient_tape/mean_squared_error/Cast(1ffffff??9ffffff??Affffff??Iffffff??a&??? ?2?i?Ż=????Unknown
?EHostReadVariableOp",sequential_15/dense_31/MatMul/ReadVariableOp(1ffffff??9ffffff??Affffff??Iffffff??a&??? ?2?i?????????Unknown
`FHostDivNoNan"
div_no_nan(1333333??9333333??A333333??I333333??a?cqy@10?i??g?????Unknown
wGHostReadVariableOp"div_no_nan/ReadVariableOp_1(1      ??9      ??A      ??I      ??a??g???*?i?Y?3P????Unknown
wHHostMul"&gradient_tape/mean_squared_error/mul_1(1      ??9      ??A      ??I      ??a??g???*?i     ???Unknown