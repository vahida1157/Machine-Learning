"?D
BHostIDLE"IDLE1    ??"AA    ??"Aa?!۟????i?!۟?????Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1??????@9??????@A??????@I??????@a׸l?Css?i??o'?????Unknown?
sHostDestroyResourceOp"DestroyResourceOp(133333?D@9????????A33333?D@I????????ai?8+|?i"??m????Unknown?
?HostResourceApplyAdam""Adam/Adam/update/ResourceApplyAdam(1ffffff=@9ffffff=@Affffff=@Iffffff=@ab?q
u??i???^?????Unknown
dHostDataset"Iterator::Model(1333333?@9333333?@A     ?:@I     ?:@a??Jo\b?i??)????Unknown
iHostWriteSummary"WriteSummary(13333335@93333335@A3333335@I3333335@a)??I??i?t7?q????Unknown?
?HostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate(1     ?1@9     ?1@A333333,@I333333,@a4?????>i??'-?????Unknown
?HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1      /@9      /@A333333*@I333333*@a?_3}!?>i2"p?????Unknown
?	HostResourceApplyAdam"$Adam/Adam/update_3/ResourceApplyAdam(1333333$@9333333$@A333333$@I333333$@a?V?>i?;??????Unknown
t
Host_FusedMatMul"sequential_7/dense_14/Relu(1333333#@9333333#@A333333#@I333333#@a݂?7?>i?$??????Unknown
?HostResourceApplyAdam"$Adam/Adam/update_2/ResourceApplyAdam(1??????!@9??????!@A??????!@I??????!@a@ſ@???>i?e?-????Unknown
~HostMatMul"*gradient_tape/sequential_7/dense_14/MatMul(1ffffff!@9ffffff!@Affffff!@Iffffff!@a0??E'e?>i??? K????Unknown
^HostGatherV2"GatherV2(1??????@9??????@A??????@I??????@a?̧x1?>i4$?$e????Unknown
?HostResourceApplyAdam"$Adam/Adam/update_1/ResourceApplyAdam(1      @9      @A      @I      @a?뤵=??>i??{????Unknown
~HostMatMul"*gradient_tape/sequential_7/dense_15/MatMul(1      @9      @A      @I      @a?c??E?>i???`?????Unknown
gHostStridedSlice"strided_slice(1??????@9??????@A??????@I??????@af?M?EB?>i5???????Unknown
eHost
LogicalAnd"
LogicalAnd(1??????@9??????@A??????@I??????@a9????>?>i?t???????Unknown?
sHostDataset"Iterator::Model::ParallelMapV2(1??????@9??????@A??????@I??????@a?5A"???>i??7??????Unknown
xHostDataset"#Iterator::Model::ParallelMapV2::Zip(1ffffffC@9ffffffC@A??????@I??????@a%U?a?>im???????Unknown
?HostDynamicStitch".gradient_tape/mean_squared_error/DynamicStitch(1ffffff@9ffffff@Affffff@Iffffff@a?T>_???>i?b??????Unknown
[HostAddV2"Adam/add(1      @9      @A      @I      @ańhi??>i?:R?????Unknown
wHost_FusedMatMul"sequential_7/dense_15/BiasAdd(1ffffff@9ffffff@Affffff@Iffffff@a???}???>i?S)?????Unknown
?HostTile"5gradient_tape/mean_squared_error/weighted_loss/Tile_1(1      @9      @A      @I      @a-t;????>i??o?????Unknown
?HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1333333@9333333@A333333@I333333@a?e????>i??Ly????Unknown
lHostIteratorGetNext"IteratorGetNext(1333333@9333333@A333333@I333333@a?e????>i"H*?????Unknown
?HostBiasAddGrad"7gradient_tape/sequential_7/dense_15/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a?c??E?>i??(????Unknown
?HostBiasAddGrad"7gradient_tape/sequential_7/dense_14/BiasAdd/BiasAddGrad(1333333@9333333@A333333@I333333@au?8?Ę?>iEn?1????Unknown
VHostSum"Sum_2(1ffffff@9ffffff@Affffff@Iffffff@aW?b????>i??Q[;????Unknown
`HostGatherV2"
GatherV2_1(1??????@9??????@A??????@I??????@a9????>?>i??zD????Unknown
?HostReluGrad",gradient_tape/sequential_7/dense_14/ReluGrad(1      @9      @A      @I      @a?R????>i???L????Unknown
YHostPow"Adam/Pow(1333333@9333333@A333333@I333333@a݂?7?>i?	U????Unknown
? HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1333333@9333333??A333333@I333333??a݂?7?>i???$]????Unknown
v!HostAssignAddVariableOp"AssignAddVariableOp_2(1??????@9??????@A??????@I??????@a@ſ@???>i?!Փd????Unknown
?"HostMatMul",gradient_tape/sequential_7/dense_15/MatMul_1(1?????? @9?????? @A?????? @I?????? @a%U?a?>i.???k????Unknown
?#HostReadVariableOp",sequential_7/dense_14/BiasAdd/ReadVariableOp(1?????? @9?????? @A?????? @I?????? @a%U?a?>isL??r????Unknown
t$HostAssignAddVariableOp"AssignAddVariableOp(1       @9       @A       @I       @ańhi??>i?敆y????Unknown
V%HostMean"Mean(1       @9       @A       @I       @ańhi??>i'??H?????Unknown
}&HostMaximum"(gradient_tape/mean_squared_error/Maximum(1ffffff??9ffffff??Affffff??Iffffff??a???}???>i? ????Unknown
v'HostAssignAddVariableOp"AssignAddVariableOp_4(1333333??9333333??A333333??I333333??a?e????>i/?\r?????Unknown
p(HostSquaredDifference"SquaredDifference(1333333??9333333??A333333??I333333??a?e????>i?s?0?????Unknown
~)HostAssignAddVariableOp"Adam/Adam/AssignAddVariableOp(1????????9????????A????????I????????a??????>iw"???????Unknown
]*HostCast"Adam/Cast_1(1????????9????????A????????I????????a??????>i&Ѫ ?????Unknown
?+HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1?????3@9?????3@A????????I????????a??????>i??h?????Unknown
V,HostCast"Cast(1      ??9      ??A      ??I      ??a?c??E?>i?3z?????Unknown
T-HostMul"Mul(1      ??9      ??A      ??I      ??a?c??E?>i]?{??????Unknown
}.HostRealDiv"(gradient_tape/mean_squared_error/truediv(1????????9????????A????????I????????a#??ʑ?>iK????????Unknown
?/HostDivNoNan"?gradient_tape/mean_squared_error/weighted_loss/value/div_no_nan(1????????9????????A????????I????????a#??ʑ?>i9caT?????Unknown
t0HostReadVariableOp"Adam/Cast/ReadVariableOp(1333333??9333333??A333333??I333333??a݂?7?>i<&Ub?????Unknown
[1HostPow"
Adam/Pow_1(1333333??9333333??A333333??I333333??a݂?7?>i??Hp?????Unknown
b2HostDivNoNan"div_no_nan_1(1333333??9333333??A333333??I333333??a݂?7?>iB?<~?????Unknown
3HostFloorDiv")gradient_tape/mean_squared_error/floordiv(1333333??9333333??A333333??I333333??a݂?7?>iEo0??????Unknown
w4HostCast"%gradient_tape/mean_squared_error/Cast(1????????9????????A????????I????????a@ſ@???>i]7?C?????Unknown
u5HostSum"$gradient_tape/mean_squared_error/Sum(1????????9????????A????????I????????a@ſ@???>iu???????Unknown
u6HostSub"$gradient_tape/mean_squared_error/sub(1????????9????????A????????I????????a@ſ@???>i?ǎ??????Unknown
v7HostAssignAddVariableOp"AssignAddVariableOp_1(1      ??9      ??A      ??I      ??ańhi??>i????????Unknown
?8HostBroadcastTo",gradient_tape/mean_squared_error/BroadcastTo(1      ??9      ??A      ??I      ??ańhi??>i?azt?????Unknown
|9HostDivNoNan"&mean_squared_error/weighted_loss/value(1      ??9      ??A      ??I      ??ańhi??>i/p??????Unknown
o:HostReadVariableOp"Adam/ReadVariableOp(1????????9????????A????????I????????aKD??S?>iV???????Unknown
v;HostAssignAddVariableOp"AssignAddVariableOp_3(1????????9????????A????????I????????aKD??S?>i??]??????Unknown
u<HostMul"$gradient_tape/mean_squared_error/Mul(1????????9????????A????????I????????aKD??S?>iڥ???????Unknown
v=HostReadVariableOp"Adam/Cast_2/ReadVariableOp(1????????9????????A????????I????????a??????>i1}̨?????Unknown
`>HostDivNoNan"
div_no_nan(1????????9????????A????????I????????a??????>i?T?\?????Unknown
??HostReadVariableOp"+sequential_7/dense_15/MatMul/ReadVariableOp(1????????9????????A????????I????????a??????>i?+??????Unknown
v@HostReadVariableOp"Adam/Cast_3/ReadVariableOp(1ffffff??9ffffff??Affffff??Iffffff??aW?b????>iK5n?????Unknown
?AHostReadVariableOp"+sequential_7/dense_14/MatMul/ReadVariableOp(1ffffff??9ffffff??Affffff??Iffffff??aW?b????>i?????????Unknown
XBHostCast"Cast_1(1333333??9333333??A333333??I333333??a݂?7?>i8Ƨ??????Unknown
aCHostIdentity"Identity(1333333??9333333??A333333??I333333??a݂?7?>i?????????Unknown?
wDHostReadVariableOp"div_no_nan/ReadVariableOp_1(1333333??9333333??A333333??I333333??a݂?7?>i:????????Unknown
yEHostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1333333??9333333??A333333??I333333??a݂?7?>i?j???????Unknown
?FHostReadVariableOp",sequential_7/dense_15/BiasAdd/ReadVariableOp(1333333??9333333??A333333??I333333??a݂?7?>i<L???????Unknown
uGHostReadVariableOp"div_no_nan/ReadVariableOp(1      ??9      ??A      ??I      ??ańhi??>i?2
??????Unknown
wHHostReadVariableOp"div_no_nan_1/ReadVariableOp(1      ??9      ??A      ??I      ??ańhi??>ij?O?????Unknown
wIHostMul"&gradient_tape/mean_squared_error/mul_1(1      ??9      ??A      ??I      ??ańhi??>i      ???Unknown*?C
uHostFlushSummaryWriter"FlushSummaryWriter(1??????@9??????@A??????@I??????@au>?a???iu>?a????Unknown?
sHostDestroyResourceOp"DestroyResourceOp(133333?D@9????????A33333?D@I????????a@?????i?o{mW???Unknown?
?HostResourceApplyAdam""Adam/Adam/update/ResourceApplyAdam(1ffffff=@9ffffff=@Affffff=@Iffffff=@a5ԩZ?x??i???P????Unknown
dHostDataset"Iterator::Model(1333333?@9333333?@A     ?:@I     ?:@a?j?_???i??B?????Unknown
iHostWriteSummary"WriteSummary(13333335@93333335@A3333335@I3333335@alJC̣z?i?E??1???Unknown?
?HostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate(1     ?1@9     ?1@A333333,@I333333,@ag?%?ηq?i???x?<???Unknown
?HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1      /@9      /@A333333*@I333333*@a?H?vp?i?"m??]???Unknown
?HostResourceApplyAdam"$Adam/Adam/update_3/ResourceApplyAdam(1333333$@9333333$@A333333$@I333333$@a?f?bi?ii?B??v???Unknown
t	Host_FusedMatMul"sequential_7/dense_14/Relu(1333333#@9333333#@A333333#@I333333#@a?9??k h?i??=????Unknown
?
HostResourceApplyAdam"$Adam/Adam/update_2/ResourceApplyAdam(1??????!@9??????!@A??????!@I??????!@a?_)??f?i=??-????Unknown
~HostMatMul"*gradient_tape/sequential_7/dense_14/MatMul(1ffffff!@9ffffff!@Affffff!@Iffffff!@aSD?za?e?iG:=W????Unknown
^HostGatherV2"GatherV2(1??????@9??????@A??????@I??????@a?3D? Zc?i{~.Xe????Unknown
?HostResourceApplyAdam"$Adam/Adam/update_1/ResourceApplyAdam(1      @9      @A      @I      @an?2?U`?ig??K?????Unknown
~HostMatMul"*gradient_tape/sequential_7/dense_15/MatMul(1      @9      @A      @I      @a-Ȭt?(^?i????????Unknown
gHostStridedSlice"strided_slice(1??????@9??????@A??????@I??????@a?$???\?i????"????Unknown
eHost
LogicalAnd"
LogicalAnd(1??????@9??????@A??????@I??????@a???y$[?i?'G?	???Unknown?
sHostDataset"Iterator::Model::ParallelMapV2(1??????@9??????@A??????@I??????@a52???W?i?@??????Unknown
xHostDataset"#Iterator::Model::ParallelMapV2::Zip(1ffffffC@9ffffffC@A??????@I??????@a??x^U?i}? ???Unknown
?HostDynamicStitch".gradient_tape/mean_squared_error/DynamicStitch(1ffffff@9ffffff@Affffff@Iffffff@a?? 6??T?iv?E?`*???Unknown
[HostAddV2"Adam/add(1      @9      @A      @I      @as??MT?i?qltn4???Unknown
wHost_FusedMatMul"sequential_7/dense_15/BiasAdd(1ffffff@9ffffff@Affffff@Iffffff@aa}?S?i???I?=???Unknown
?HostTile"5gradient_tape/mean_squared_error/weighted_loss/Tile_1(1      @9      @A      @I      @a?tģ?Q?i??F???Unknown
?HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1333333@9333333@A333333@I333333@a<>???Q?i???RO???Unknown
lHostIteratorGetNext"IteratorGetNext(1333333@9333333@A333333@I333333@a<>???Q?i??h?W???Unknown
?HostBiasAddGrad"7gradient_tape/sequential_7/dense_15/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a-Ȭt?(N?i??4h_???Unknown
?HostBiasAddGrad"7gradient_tape/sequential_7/dense_14/BiasAdd/BiasAddGrad(1333333@9333333@A333333@I333333@a[??,'M?i?.??f???Unknown
VHostSum"Sum_2(1ffffff@9ffffff@Affffff@Iffffff@a?K??%L?i??s?m???Unknown
`HostGatherV2"
GatherV2_1(1??????@9??????@A??????@I??????@a???y$K?i?`$??t???Unknown
?HostReluGrad",gradient_tape/sequential_7/dense_14/ReluGrad(1      @9      @A      @I      @aЦ:a?!I?i??|?z???Unknown
YHostPow"Adam/Pow(1333333@9333333@A333333@I333333@a?9??k H?i?`Հ???Unknown
?HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1333333@9333333??A333333@I333333??a?9??k H?i??D9݆???Unknown
v HostAssignAddVariableOp"AssignAddVariableOp_2(1??????@9??????@A??????@I??????@a?_)??F?i ?@?d????Unknown
?!HostMatMul",gradient_tape/sequential_7/dense_15/MatMul_1(1?????? @9?????? @A?????? @I?????? @a??x^E?i=]Ⱦ?????Unknown
?"HostReadVariableOp",sequential_7/dense_14/BiasAdd/ReadVariableOp(1?????? @9?????? @A?????? @I?????? @a??x^E?iz?O??????Unknown
t#HostAssignAddVariableOp"AssignAddVariableOp(1       @9       @A       @I       @as??MD?i?mc??????Unknown
V$HostMean"Mean(1       @9       @A       @I       @as??MD?i??vX ????Unknown
}%HostMaximum"(gradient_tape/mean_squared_error/Maximum(1ffffff??9ffffff??Affffff??Iffffff??aa}?C?i?%?ƥ???Unknown
v&HostAssignAddVariableOp"AssignAddVariableOp_4(1333333??9333333??A333333??I333333??a<>???A?i?̀????Unknown
p'HostSquaredDifference"SquaredDifference(1333333??9333333??A333333??I333333??a<>???A?ib?>R????Unknown
~(HostAssignAddVariableOp"Adam/Adam/AssignAddVariableOp(1????????9????????A????????I????????a)??@?i?ƥW????Unknown
])HostCast"Adam/Cast_1(1????????9????????A????????I????????a)??@?iʄ	]????Unknown
?*HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1?????3@9?????3@A????????I????????a)??@?i~FLtb????Unknown
V+HostCast"Cast(1      ??9      ??A      ??I      ??a-Ȭt?(>?i??'????Unknown
T,HostMul"Mul(1      ??9      ??A      ??I      ??a-Ȭt?(>?i?q???????Unknown
}-HostRealDiv"(gradient_tape/mean_squared_error/truediv(1????????9????????A????????I????????a??1#:?i???0????Unknown
?.HostDivNoNan"?gradient_tape/mean_squared_error/weighted_loss/value/div_no_nan(1????????9????????A????????I????????a??1#:?it??]u????Unknown
t/HostReadVariableOp"Adam/Cast/ReadVariableOp(1333333??9333333??A333333??I333333??a?9??k 8?i??'ky????Unknown
[0HostPow"
Adam/Pow_1(1333333??9333333??A333333??I333333??a?9??k 8?i?x}????Unknown
b1HostDivNoNan"div_no_nan_1(1333333??9333333??A333333??I333333??a?9??k 8?iI ??????Unknown
2HostFloorDiv")gradient_tape/mean_squared_error/floordiv(1333333??9333333??A333333??I333333??a?9??k 8?i?1~??????Unknown
w3HostCast"%gradient_tape/mean_squared_error/Cast(1????????9????????A????????I????????a?_)??6?i?|JI????Unknown
u4HostSum"$gradient_tape/mean_squared_error/Sum(1????????9????????A????????I????????a?_)??6?i??y????Unknown
u5HostSub"$gradient_tape/mean_squared_error/sub(1????????9????????A????????I????????a?_)??6?i?w??????Unknown
v6HostAssignAddVariableOp"AssignAddVariableOp_1(1      ??9      ??A      ??I      ??as??M4?i%?T????Unknown
?7HostBroadcastTo",gradient_tape/mean_squared_error/BroadcastTo(1      ??9      ??A      ??I      ??as??M4?i6S?y?????Unknown
|8HostDivNoNan"&mean_squared_error/weighted_loss/value(1      ??9      ??A      ??I      ??as??M4?iG?Z????Unknown
o9HostReadVariableOp"Adam/ReadVariableOp(1????????9????????A????????I????????aN?g?P2?i<?*??????Unknown
v:HostAssignAddVariableOp"AssignAddVariableOp_3(1????????9????????A????????I????????aN?g?P2?i1&@??????Unknown
u;HostMul"$gradient_tape/mean_squared_error/Mul(1????????9????????A????????I????????aN?g?P2?i&?U?#????Unknown
v<HostReadVariableOp"Adam/Cast_2/ReadVariableOp(1????????9????????A????????I????????a)??0?i ??&????Unknown
`=HostDivNoNan"
div_no_nan(1????????9????????A????????I????????a)??0?i?t?_)????Unknown
?>HostReadVariableOp"+sequential_7/dense_15/MatMul/ReadVariableOp(1????????9????????A????????I????????a)??0?i??9,????Unknown
v?HostReadVariableOp"Adam/Cast_3/ReadVariableOp(1ffffff??9ffffff??Affffff??Iffffff??a?K??%,?is
gp?????Unknown
?@HostReadVariableOp"+sequential_7/dense_14/MatMul/ReadVariableOp(1ffffff??9ffffff??Affffff??Iffffff??a?K??%,?i2??Ͱ????Unknown
XAHostCast"Cast_1(1333333??9333333??A333333??I333333??a?9??k (?i?GM?2????Unknown
aBHostIdentity"Identity(1333333??9333333??A333333??I333333??a?9??k (?izP۴????Unknown?
wCHostReadVariableOp"div_no_nan/ReadVariableOp_1(1333333??9333333??A333333??I333333??a?9??k (?iY??6????Unknown
yDHostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1333333??9333333??A333333??I333333??a?9??k (?i?ax??????Unknown
?EHostReadVariableOp",sequential_7/dense_15/BiasAdd/ReadVariableOp(1333333??9333333??A333333??I333333??a?9??k (?ifj1?:????Unknown
uFHostReadVariableOp"div_no_nan/ReadVariableOp(1      ??9      ??A      ??I      ??as??M$?i?Fv?|????Unknown
wGHostReadVariableOp"div_no_nan_1/ReadVariableOp(1      ??9      ??A      ??I      ??as??M$?iv#?O?????Unknown
wHHostMul"&gradient_tape/mean_squared_error/mul_1(1      ??9      ??A      ??I      ??as??M$?i?????????Unknown