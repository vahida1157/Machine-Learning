	??d?`T????d?`T??!??d?`T??	4}???&@4}???&@!4}???&@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$??d?`T??ı.n???Ac?=yX??Y$????ۧ?*	gffff&R@2F
Iterator::Model?p=
ף??!KBfbF@)B>?٬???1	?5?}C@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate??y?):??!??w???8@)??H?}??1y?r???3@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat2U0*???!?:G??5@)???<,Ԋ?1hD??H2@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip??#?????!??????K@)n??t?1?+?uK?@:Preprocessing2U
Iterator::Model::ParallelMapV2?J?4q?!J???#@)?J?4q?1J???#@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice_?Q?k?!G????@)_?Q?k?1G????@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??_?Le?!?Omo?@)??_?Le?1?Omo?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap+??????!??zv?:@)_?Q?[?1G????@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 11.3% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*high2t26.6 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.94}???&@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	ı.n???ı.n???!ı.n???      ??!       "      ??!       *      ??!       2	c?=yX??c?=yX??!c?=yX??:      ??!       B      ??!       J	$????ۧ?$????ۧ?!$????ۧ?R      ??!       Z	$????ۧ?$????ۧ?!$????ۧ?JCPU_ONLYY4}???&@b 