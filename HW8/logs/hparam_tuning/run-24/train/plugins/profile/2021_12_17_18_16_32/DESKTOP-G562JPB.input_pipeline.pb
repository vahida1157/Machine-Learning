	[????<??[????<??![????<??	vnK?|#@vnK?|#@!vnK?|#@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$[????<??F%u???AyX?5?;??Yj?q?????*	     ?R@2F
Iterator::ModelM??St$??!"""""">@)?J?4??1gfffff6@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat???&??!??????8@)vq?-??15@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip???<,Ԫ?!wwwwwwQ@)2??%䃎?1??????3@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate???߾??!DDDDDD2@)??~j?t??1VUUUUU)@:Preprocessing2U
Iterator::Model::ParallelMapV2?????w?!??????@)?????w?1??????@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapU???N@??!9@)??ZӼ?t?1433333@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?J?4q?!gfffff@)?J?4q?1gfffff@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?????g?!??????@)?????g?1??????@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 9.7% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*high2t27.9 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9vnK?|#@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	F%u???F%u???!F%u???      ??!       "      ??!       *      ??!       2	yX?5?;??yX?5?;??!yX?5?;??:      ??!       B      ??!       J	j?q?????j?q?????!j?q?????R      ??!       Z	j?q?????j?q?????!j?q?????JCPU_ONLYYvnK?|#@b 