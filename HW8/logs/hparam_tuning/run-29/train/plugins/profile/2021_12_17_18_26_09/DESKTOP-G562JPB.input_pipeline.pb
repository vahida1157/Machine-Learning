	?h o????h o???!?h o???	[?W??y @[?W??y @![?W??y @"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?h o???=?U?????A?46<??Y??ܵ?|??*	??????H@2F
Iterator::Model???&??!??n??B@)?W[?????1???a>@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat???H??!??????@)???S㥋?1????);@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateM?O???!???jR4@)?<,Ԛ?}?1??jR`-@:Preprocessing2U
Iterator::Model::ParallelMapV2??H?}m?!?W?M?@)??H?}m?1?W?M?@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipX9??v???!jA?F?/O@)??H?}m?1?W?M?@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceǺ???f?!!?
??@)Ǻ???f?1!?
??@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensora2U0*?c?!@????P@)a2U0*?c?1@????P@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapZd;?O???!??1??#7@)Ǻ???V?1!?
??@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 8.2% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*high2t33.3 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9[?W??y @>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	=?U?????=?U?????!=?U?????      ??!       "      ??!       *      ??!       2	?46<???46<??!?46<??:      ??!       B      ??!       J	??ܵ?|????ܵ?|??!??ܵ?|??R      ??!       Z	??ܵ?|????ܵ?|??!??ܵ?|??JCPU_ONLYY[?W??y @b 