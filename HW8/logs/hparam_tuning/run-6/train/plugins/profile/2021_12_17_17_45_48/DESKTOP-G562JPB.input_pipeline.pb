	M?O???M?O???!M?O???	?Q`?i+@?Q`?i+@!?Q`?i+@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$M?O????Fx$??A??k	????Y?lV}???*	    ?Q@2F
Iterator::Model?q??????!?$I?$IF@)?A`??"??1۶m۶?B@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?Q?????!      9@)?!??u???1I?$I?$4@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatX9??v???!I?$I?$6@)???<,Ԋ?1n۶m۶2@:Preprocessing2U
Iterator::Model::ParallelMapV2U???N@s?!?m۶m?@)U???N@s?1?m۶m?@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?j+??ݣ?!n۶m۶K@)?J?4q?1      @:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice_?Q?k?!۶m۶m@)_?Q?k?1۶m۶m@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensora2U0*?c?!ܶm۶m@)a2U0*?c?1ܶm۶m@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapjM????!?$I?$I;@)-C??6Z?1?$I?$I@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 13.7% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*high2t29.2 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9?Q`?i+@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?Fx$???Fx$??!?Fx$??      ??!       "      ??!       *      ??!       2	??k	??????k	????!??k	????:      ??!       B      ??!       J	?lV}????lV}???!?lV}???R      ??!       Z	?lV}????lV}???!?lV}???JCPU_ONLYY?Q`?i+@b 