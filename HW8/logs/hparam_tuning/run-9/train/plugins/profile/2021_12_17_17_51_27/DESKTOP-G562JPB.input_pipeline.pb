	q=
ףp??q=
ףp??!q=
ףp??	*g???)@*g???)@!*g???)@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$q=
ףp??5?8EGr??A,Ԛ????Yj?t???*	     ?Q@2F
Iterator::ModelV-???!?)͋??D@)=?U?????1???S?A@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenatee?X???!Ni^??8@)??Pk?w??1k:`?3@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatX?5?;N??!P$?Ҽ?7@)_?Qڋ?1͋??pJ3@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipsh??|???!=?2t?nM@)??ZӼ?t?1?Q?٨?@:Preprocessing2U
Iterator::Model::ParallelMapV2n??t?!0?D?)?@)n??t?10?D?)?@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice_?Q?k?!͋??pJ@)_?Q?k?1͋??pJ@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorF%u?k?!br1?@)F%u?k?1br1?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap+??????!??Q?٨;@)/n??b?1d-C??@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 12.8% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*high2t29.7 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9*g???)@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	5?8EGr??5?8EGr??!5?8EGr??      ??!       "      ??!       *      ??!       2	,Ԛ????,Ԛ????!,Ԛ????:      ??!       B      ??!       J	j?t???j?t???!j?t???R      ??!       Z	j?t???j?t???!j?t???JCPU_ONLYY*g???)@b 