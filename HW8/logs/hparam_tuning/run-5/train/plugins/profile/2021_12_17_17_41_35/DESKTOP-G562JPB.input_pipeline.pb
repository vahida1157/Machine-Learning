	0L?
F%??0L?
F%??!0L?
F%??	?שFf$@?שFf$@!?שFf$@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$0L?
F%??NbX9???A?`TR'???Y7?[ A??*	????̌K@2F
Iterator::Model??ZӼ???!?XW?B@)??ǘ????1,əíf=@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?? ?rh??!<???]?>@)?ZӼ???1?؋u?9@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateZd;?O???!֬?@?4@)?? ?rh??1<???]?.@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip??W?2ġ?!B????|O@)"??u??q?1@1z?I7@:Preprocessing2U
Iterator::Model::ParallelMapV2?J?4q?!8?S?q}@)?J?4q?18?S?q}@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?~j?t?h?!???NQ?@)?~j?t?h?1???NQ?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorǺ???f?!??78?S@)Ǻ???f?1??78?S@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapS?!?uq??!?؋u?Q8@)ŏ1w-!_?1a???@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 10.2% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*high2t29.2 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9?שFf$@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	NbX9???NbX9???!NbX9???      ??!       "      ??!       *      ??!       2	?`TR'????`TR'???!?`TR'???:      ??!       B      ??!       J	7?[ A??7?[ A??!7?[ A??R      ??!       Z	7?[ A??7?[ A??!7?[ A??JCPU_ONLYY?שFf$@b 