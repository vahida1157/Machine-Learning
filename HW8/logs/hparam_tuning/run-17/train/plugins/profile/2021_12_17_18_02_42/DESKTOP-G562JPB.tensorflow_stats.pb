"?C
BHostIDLE"IDLE1fffff?@Afffff?@anC??????inC???????Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1     B?@9     B?@A     B?@I     B?@a?G???k??iBg???\???Unknown?
?HostResourceApplyAdam""Adam/Adam/update/ResourceApplyAdam(133333sB@933333sB@A33333sB@I33333sB@a'?˛i?w?ih??Z????Unknown
dHostDataset"Iterator::Model(133333sA@933333sA@Affffff;@Iffffff;@a<?_?q?i???d?????Unknown
iHostWriteSummary"WriteSummary(1??????4@9??????4@A??????4@I??????4@akI'3?qj?i)?ra?????Unknown?
?HostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate(1??????6@9??????6@A??????2@I??????2@a??K?o"h?i?+9?????Unknown
?HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1ffffff3@9ffffff3@A?????0@I?????0@aI??d?i?-???????Unknown
uHost_FusedMatMul"sequential_18/dense_36/Relu(1333333.@9333333.@A333333.@I333333.@aAkݻwbc?iR?d)
???Unknown
?	HostResourceApplyAdam"$Adam/Adam/update_2/ResourceApplyAdam(1ffffff(@9ffffff(@Affffff(@Iffffff(@a?[Ln?R_?i?1??????Unknown
?
HostResourceApplyAdam"$Adam/Adam/update_3/ResourceApplyAdam(1      (@9      (@A      (@I      (@a??p?]?^?i???|:)???Unknown
^HostGatherV2"GatherV2(1      #@9      #@A      #@I      #@a?\?'*dX?i?Ƹ?l5???Unknown
?HostResourceApplyAdam"$Adam/Adam/update_1/ResourceApplyAdam(1ffffff!@9ffffff!@Affffff!@Iffffff!@a?oKWVV?iV?F??@???Unknown
HostMatMul"+gradient_tape/sequential_18/dense_36/MatMul(1333333!@9333333!@A333333!@I333333!@a?ݺ?V?i/[??K???Unknown
sHostDataset"Iterator::Model::ParallelMapV2(1      @9      @A      @I      @as?&??AS?iu???BU???Unknown
HostMatMul"+gradient_tape/sequential_18/dense_37/MatMul(1      @9      @A      @I      @as?&??AS?i??/??^???Unknown
[HostAddV2"Adam/add(1??????@9??????@A??????@I??????@a?fo[?nP?in9]?g???Unknown
gHostStridedSlice"strided_slice(1ffffff@9ffffff@Affffff@Iffffff@a?[Ln?RO?i?????n???Unknown
xHost_FusedMatMul"sequential_18/dense_37/BiasAdd(1333333@9333333@A333333@I333333@aK??%t?M?i ;?av???Unknown
eHost
LogicalAnd"
LogicalAnd(1??????@9??????@A??????@I??????@ao?b?DM?i????}???Unknown?
xHostDataset"#Iterator::Model::ParallelMapV2::Zip(1     ?H@9     ?H@A??????@I??????@as?K??K?i??!??????Unknown
lHostIteratorGetNext"IteratorGetNext(1??????@9??????@A??????@I??????@a????J?iݪ?N????Unknown
?HostDynamicStitch".gradient_tape/mean_squared_error/DynamicStitch(1      @9      @A      @I      @a??άI?iab?͹????Unknown
?HostTile"5gradient_tape/mean_squared_error/weighted_loss/Tile_1(1333333@9333333@A333333@I333333@aD?o??D?iR>??????Unknown
?HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1ffffff@9ffffff@Affffff@Iffffff@aJ??T?C?iec1U?????Unknown
VHostSum"Sum_2(1??????@9??????@A??????@I??????@a?θ)??B?i??;M\????Unknown
?HostBiasAddGrad"8gradient_tape/sequential_18/dense_37/BiasAdd/BiasAddGrad(1??????@9??????@A??????@I??????@a?S?fk|B?i??h?????Unknown
?HostBiasAddGrad"8gradient_tape/sequential_18/dense_36/BiasAdd/BiasAddGrad(1333333@9333333@A333333@I333333@a5]&??uA?i?ҍ?X????Unknown
YHostPow"Adam/Pow(1ffffff
@9ffffff
@Affffff
@Iffffff
@a??J?@?i=e?K?????Unknown
?HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1ffffff
@9ffffff??Affffff
@Iffffff??a??J?@?i???ѱ???Unknown
?HostReluGrad"-gradient_tape/sequential_18/dense_36/ReluGrad(1ffffff
@9ffffff
@Affffff
@Iffffff
@a??J?@?i??dR????Unknown
?HostReadVariableOp"-sequential_18/dense_37/BiasAdd/ReadVariableOp(1ffffff
@9ffffff
@Affffff
@Iffffff
@a??J?@?ie??J????Unknown
` HostGatherV2"
GatherV2_1(1??????@9??????@A??????@I??????@a%?'1G???i`B??E????Unknown
[!HostPow"
Adam/Pow_1(1333333@9333333@A333333@I333333@aK??%t?=?i????????Unknown
?"HostMatMul"-gradient_tape/sequential_18/dense_37/MatMul_1(1ffffff@9ffffff@Affffff@Iffffff@a?????<?i??jޖ????Unknown
v#HostAssignAddVariableOp"AssignAddVariableOp_2(1??????@9??????@A??????@I??????@as?K??;?i{C?2????Unknown
V$HostMean"Mean(1??????@9??????@A??????@I??????@as?K??;?i?????????Unknown
o%HostReadVariableOp"Adam/ReadVariableOp(1333333@9333333@A333333@I333333@a-'???8?iޱBC?????Unknown
p&HostSquaredDifference"SquaredDifference(1??????@9??????@A??????@I??????@aT-?}?6?iirEm????Unknown
]'HostCast"Adam/Cast_1(1?????? @9?????? @A?????? @I?????? @a?6?'?5?iKiqj????Unknown
?(HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(133333?8@933333?8@A?????? @I?????? @a?6?'?5?i?ip??????Unknown
})HostMaximum"(gradient_tape/mean_squared_error/Maximum(1       @9       @A       @I       @a{@Kr>?4?i??>?b????Unknown
V*HostCast"Cast(1????????9????????A????????I????????a?S?fk|2?i???d?????Unknown
~+HostAssignAddVariableOp"Adam/Adam/AssignAddVariableOp(1ffffff??9ffffff??Affffff??Iffffff??a?????,?iӎU}~????Unknown
t,HostAssignAddVariableOp"AssignAddVariableOp(1ffffff??9ffffff??Affffff??Iffffff??a?????,?i???J????Unknown
v-HostAssignAddVariableOp"AssignAddVariableOp_4(1ffffff??9ffffff??Affffff??Iffffff??a?????,?i1???????Unknown
|.HostDivNoNan"&mean_squared_error/weighted_loss/value(1ffffff??9ffffff??Affffff??Iffffff??a?????,?i`?S??????Unknown
v/HostReadVariableOp"Adam/Cast_3/ReadVariableOp(1????????9????????A????????I????????a????*?i????????Unknown
T0HostMul"Mul(1????????9????????A????????I????????a????*?i "F>9????Unknown
1HostFloorDiv")gradient_tape/mean_squared_error/floordiv(1????????9????????A????????I????????a????*?iPk?y?????Unknown
}2HostRealDiv"(gradient_tape/mean_squared_error/truediv(1????????9????????A????????I????????a????*?i??8??????Unknown
u3HostSub"$gradient_tape/mean_squared_error/sub(1????????9????????A????????I????????aT-?}?&?i3?P6?????Unknown
?4HostReadVariableOp",sequential_18/dense_37/MatMul/ReadVariableOp(1????????9????????A????????I????????aT-?}?&?i?kh?b????Unknown
t5HostReadVariableOp"Adam/Cast/ReadVariableOp(1      ??9      ??A      ??I      ??a{@Kr>?$?iz?O[?????Unknown
?6HostBroadcastTo",gradient_tape/mean_squared_error/BroadcastTo(1      ??9      ??A      ??I      ??a{@Kr>?$?i.?6??????Unknown
v7HostReadVariableOp"Adam/Cast_2/ReadVariableOp(1????????9????????A????????I????????a?S?fk|"?i#??????Unknown
v8HostAssignAddVariableOp"AssignAddVariableOp_1(1????????9????????A????????I????????a?S?fk|"?iؐ??C????Unknown
b9HostDivNoNan"div_no_nan_1(1????????9????????A????????I????????a?S?fk|"?i??YSk????Unknown
u:HostMul"$gradient_tape/mean_squared_error/Mul(1????????9????????A????????I????????a?S?fk|"?i?l?????Unknown
u;HostSum"$gradient_tape/mean_squared_error/Sum(1????????9????????A????????I????????a?S?fk|"?iW????????Unknown
?<HostReadVariableOp",sequential_18/dense_36/MatMul/ReadVariableOp(1????????9????????A????????I????????a?S?fk|"?i,H}??????Unknown
w=HostCast"%gradient_tape/mean_squared_error/Cast(1????????9????????A????????I????????a?fo[?n ?i"???????Unknown
?>HostReadVariableOp"-sequential_18/dense_36/BiasAdd/ReadVariableOp(1????????9????????A????????I????????a?fo[?n ?i??z?????Unknown
v?HostAssignAddVariableOp"AssignAddVariableOp_3(1ffffff??9ffffff??Affffff??Iffffff??a??????i0?݆?????Unknown
`@HostDivNoNan"
div_no_nan(1ffffff??9ffffff??Affffff??Iffffff??a??????iH?2??????Unknown
?AHostDivNoNan"?gradient_tape/mean_squared_error/weighted_loss/value/div_no_nan(1ffffff??9ffffff??Affffff??Iffffff??a??????i`????????Unknown
aBHostIdentity"Identity(1333333??9333333??A333333??I333333??a-'????i????g????Unknown?
XCHostCast"Cast_1(1      ??9      ??A      ??I      ??a{@Kr>??i??? ????Unknown
wDHostReadVariableOp"div_no_nan/ReadVariableOp_1(1      ??9      ??A      ??I      ??a{@Kr>??iM$?r?????Unknown
yEHostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1      ??9      ??A      ??I      ??a{@Kr>??i????T????Unknown
wFHostMul"&gradient_tape/mean_squared_error/mul_1(1      ??9      ??A      ??I      ??a{@Kr>??iIz?????Unknown
uGHostReadVariableOp"div_no_nan/ReadVariableOp(1????????9????????A????????I????????a?fo[?n?i|$=?|????Unknown
wHHostReadVariableOp"div_no_nan_1/ReadVariableOp(1????????9????????A????????I????????a?fo[?n?i?????????Unknown*?B
uHostFlushSummaryWriter"FlushSummaryWriter(1     B?@9     B?@A     B?@I     B?@a?I?pcb??i?I?pcb???Unknown?
?HostResourceApplyAdam""Adam/Adam/update/ResourceApplyAdam(133333sB@933333sB@A33333sB@I33333sB@a?GkQ????i?R??????Unknown
dHostDataset"Iterator::Model(133333sA@933333sA@Affffff;@Iffffff;@a?D̈?iK伻)K???Unknown
iHostWriteSummary"WriteSummary(1??????4@9??????4@A??????4@I??????4@a??Ȥ??i???޼????Unknown?
?HostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate(1??????6@9??????6@A??????2@I??????2@a???????i?????????Unknown
?HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1ffffff3@9ffffff3@A?????0@I?????0@a$`??^$}?ir?????Unknown
uHost_FusedMatMul"sequential_18/dense_36/Relu(1333333.@9333333.@A333333.@I333333.@aW?;s?T{?i??Г?J???Unknown
?HostResourceApplyAdam"$Adam/Adam/update_2/ResourceApplyAdam(1ffffff(@9ffffff(@Affffff(@Iffffff(@aS?M3v?i(8l??v???Unknown
?	HostResourceApplyAdam"$Adam/Adam/update_3/ResourceApplyAdam(1      (@9      (@A      (@I      (@a???e??u?iS?7Z????Unknown
^
HostGatherV2"GatherV2(1      #@9      #@A      #@I      #@a?2q?iU?X1?????Unknown
?HostResourceApplyAdam"$Adam/Adam/update_1/ResourceApplyAdam(1ffffff!@9ffffff!@Affffff!@Iffffff!@a?U??~o?i!'9?<????Unknown
HostMatMul"+gradient_tape/sequential_18/dense_36/MatMul(1333333!@9333333!@A333333!@I333333!@aPE=?"o?ifd1
_???Unknown
sHostDataset"Iterator::Model::ParallelMapV2(1      @9      @A      @I      @a?z/??&k?i??0?????Unknown
HostMatMul"+gradient_tape/sequential_18/dense_37/MatMul(1      @9      @A      @I      @a?z/??&k?i\?/Z?9???Unknown
[HostAddV2"Adam/add(1??????@9??????@A??????@I??????@a??!:+g?i?5??P???Unknown
gHostStridedSlice"strided_slice(1ffffff@9ffffff@Affffff@Iffffff@aS?M3f?i)????f???Unknown
xHost_FusedMatMul"sequential_18/dense_37/BiasAdd(1333333@9333333@A333333@I333333@a???,?d?i?K??{???Unknown
eHost
LogicalAnd"
LogicalAnd(1??????@9??????@A??????@I??????@aJv??d?i???s?????Unknown?
xHostDataset"#Iterator::Model::ParallelMapV2::Zip(1     ?H@9     ?H@A??????@I??????@am,?x?c?i???????Unknown
lHostIteratorGetNext"IteratorGetNext(1??????@9??????@A??????@I??????@a~_?$?b?i~???????Unknown
?HostDynamicStitch".gradient_tape/mean_squared_error/DynamicStitch(1      @9      @A      @I      @a?Q?T?b?iг5?????Unknown
?HostTile"5gradient_tape/mean_squared_error/weighted_loss/Tile_1(1333333@9333333@A333333@I333333@a=y??<\?iXp?(&????Unknown
?HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1ffffff@9ffffff@Affffff@Iffffff@a?H?T?[?iY??????Unknown
VHostSum"Sum_2(1??????@9??????@A??????@I??????@a3???Z?iӟ??L????Unknown
?HostBiasAddGrad"8gradient_tape/sequential_18/dense_37/BiasAdd/BiasAddGrad(1??????@9??????@A??????@I??????@a???F?Z?i?4!U????Unknown
?HostBiasAddGrad"8gradient_tape/sequential_18/dense_36/BiasAdd/BiasAddGrad(1333333@9333333@A333333@I333333@a?˃???X?i?T????Unknown
YHostPow"Adam/Pow(1ffffff
@9ffffff
@Affffff
@Iffffff
@a?R֓?W?i~?a????Unknown
?HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1ffffff
@9ffffff??Affffff
@Iffffff??a?R֓?W?ij?ݫ?#???Unknown
?HostReluGrad"-gradient_tape/sequential_18/dense_36/ReluGrad(1ffffff
@9ffffff
@Affffff
@Iffffff
@a?R֓?W?i????z/???Unknown
?HostReadVariableOp"-sequential_18/dense_37/BiasAdd/ReadVariableOp(1ffffff
@9ffffff
@Affffff
@Iffffff
@a?R֓?W?i(???m;???Unknown
`HostGatherV2"
GatherV2_1(1??????@9??????@A??????@I??????@a??5?qV?iz??/?F???Unknown
[ HostPow"
Adam/Pow_1(1333333@9333333@A333333@I333333@a???,?T?i???%Q???Unknown
?!HostMatMul"-gradient_tape/sequential_18/dense_37/MatMul_1(1ffffff@9ffffff@Affffff@Iffffff@a?z]??ET?i{h|?H[???Unknown
v"HostAssignAddVariableOp"AssignAddVariableOp_2(1??????@9??????@A??????@I??????@am,?x?S?i????e???Unknown
V#HostMean"Mean(1??????@9??????@A??????@I??????@am,?x?S?i??q(?n???Unknown
o$HostReadVariableOp"Adam/ReadVariableOp(1333333@9333333@A333333@I333333@atD??k`Q?i??3^?w???Unknown
p%HostSquaredDifference"SquaredDifference(1??????@9??????@A??????@I??????@a?Rn?o?O?i ?%:|???Unknown
]&HostCast"Adam/Cast_1(1?????? @9?????? @A?????? @I?????? @a?7(?hN?i. 0i????Unknown
?'HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(133333?8@933333?8@A?????? @I?????? @a?7(?hN?i<:??????Unknown
}(HostMaximum"(gradient_tape/mean_squared_error/Maximum(1       @9       @A       @I       @a????L?i??[?????Unknown
V)HostCast"Cast(1????????9????????A????????I????????a???F?J?i=??Br????Unknown
~*HostAssignAddVariableOp"Adam/Adam/AssignAddVariableOp(1ffffff??9ffffff??Affffff??Iffffff??a?z]??ED?i??^??????Unknown
t+HostAssignAddVariableOp"AssignAddVariableOp(1ffffff??9ffffff??Affffff??Iffffff??a?z]??ED?i?U,?????Unknown
v,HostAssignAddVariableOp"AssignAddVariableOp_4(1ffffff??9ffffff??Affffff??Iffffff??a?z]??ED?iZ????????Unknown
|-HostDivNoNan"&mean_squared_error/weighted_loss/value(1ffffff??9ffffff??Affffff??Iffffff??a?z]??ED?i?s?????Unknown
v.HostReadVariableOp"Adam/Cast_3/ReadVariableOp(1????????9????????A????????I????????a~_?$?B?i?C<?l????Unknown
T/HostMul"Mul(1????????9????????A????????I????????a~_?$?B?ii??!????Unknown
0HostFloorDiv")gradient_tape/mean_squared_error/floordiv(1????????9????????A????????I????????a~_?$?B?iA??l־???Unknown
}1HostRealDiv"(gradient_tape/mean_squared_error/truediv(1????????9????????A????????I????????a~_?$?B?i ?4?????Unknown
u2HostSub"$gradient_tape/mean_squared_error/sub(1????????9????????A????????I????????a?Rn?o???i????????Unknown
?3HostReadVariableOp",sequential_18/dense_37/MatMul/ReadVariableOp(1????????9????????A????????I????????a?Rn?o???i???????Unknown
t4HostReadVariableOp"Adam/Cast/ReadVariableOp(1      ??9      ??A      ??I      ??a????<?i??? ????Unknown
?5HostBroadcastTo",gradient_tape/mean_squared_error/BroadcastTo(1      ??9      ??A      ??I      ??a????<?i5???????Unknown
v6HostReadVariableOp"Adam/Cast_2/ReadVariableOp(1????????9????????A????????I????????a???F?:?i??Ԧ????Unknown
v7HostAssignAddVariableOp"AssignAddVariableOp_1(1????????9????????A????????I????????a???F?:?i????C????Unknown
b8HostDivNoNan"div_no_nan_1(1????????9????????A????????I????????a???F?:?il?&υ????Unknown
u9HostMul"$gradient_tape/mean_squared_error/Mul(1????????9????????A????????I????????a???F?:?i)yO??????Unknown
u:HostSum"$gradient_tape/mean_squared_error/Sum(1????????9????????A????????I????????a???F?:?i?Ux?	????Unknown
?;HostReadVariableOp",sequential_18/dense_36/MatMul/ReadVariableOp(1????????9????????A????????I????????a???F?:?i?2?L????Unknown
w<HostCast"%gradient_tape/mean_squared_error/Cast(1????????9????????A????????I????????a??!:+7?i???r1????Unknown
?=HostReadVariableOp"-sequential_18/dense_36/BiasAdd/ReadVariableOp(1????????9????????A????????I????????a??!:+7?i?"?????Unknown
v>HostAssignAddVariableOp"AssignAddVariableOp_3(1ffffff??9ffffff??Affffff??Iffffff??a?z]??E4?i?f{??????Unknown
`?HostDivNoNan"
div_no_nan(1ffffff??9ffffff??Affffff??Iffffff??a?z]??E4?im?N(????Unknown
?@HostDivNoNan"?gradient_tape/mean_squared_error/weighted_loss/value/div_no_nan(1ffffff??9ffffff??Affffff??Iffffff??a?z]??E4?i?,	?????Unknown
aAHostIdentity"Identity(1333333??9333333??A333333??I333333??atD??k`1?iEQ??????Unknown?
XBHostCast"Cast_1(1      ??9      ??A      ??I      ??a????,?i??%w?????Unknown
wCHostReadVariableOp"div_no_nan/ReadVariableOp_1(1      ??9      ??A      ??I      ??a????,?i?F??{????Unknown
yDHostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1      ??9      ??A      ??I      ??a????,?i+?68K????Unknown
wEHostMul"&gradient_tape/mean_squared_error/mul_1(1      ??9      ??A      ??I      ??a????,?i?;??????Unknown
uFHostReadVariableOp"div_no_nan/ReadVariableOp(1????????9????????A????????I????????a??!:+'?i??_L?????Unknown
wGHostReadVariableOp"div_no_nan_1/ReadVariableOp(1????????9????????A????????I????????a??!:+'?i     ???Unknown