using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CNTK;
using static CNTK.CNTKLib;

namespace CntkSerializerIssue
{
    class Program
    {
        const string FeaturesName = "features";
        const string LabelsName = "labels";
        const string TargetsName = "targets";

        static void Main(string[] args)
        {
            var d = DeviceDescriptor.UseDefaultDevice(); //DeviceDescriptor.CPUDevice;

            var channelNameToMapFilePath = new Dictionary<string, string>
            {
                { "Channel1", "" },
                { "Channel2", "" },
                { "Channel3", "" },
                { "Channel4", "" },
            };

            var ctfFilePath = "";
            var outputShape = 3;
            var maxSweeps = int.MaxValue;
            uint minibatchSize = 32;

            var source = CreateTrainMinibatchSource(channelNameToMapFilePath, ctfFilePath, outputShape, maxSweeps);

            while (true)
            {
                var minibatchData = source.GetNextMinibatch(minibatchSize, d);

                // Stop training once max epochs is reached.
                if (minibatchData.empty())
                {
                    break;
                }
            }
        }

        static MinibatchSource CreateTrainMinibatchSource(
            IReadOnlyDictionary<string, string> channelNameToMapFilePath, string ctfFilePath,
            int outputShape, int maxEpochsOrSweeps)
        {
            var imageDeserializers = channelNameToMapFilePath.Select(p =>
            {
                var deserializer = CNTKLib.ImageDeserializer(p.Value, p.Key + LabelsName, (uint)1, p.Key + FeaturesName);
                AddGrayScaleTrue(deserializer);
                return deserializer;
            }).ToArray();

            var targetCtfDeserializer = CreateCtfDeserializer(ctfFilePath, outputShape);
            var deserializerList = new List<CNTKDictionary>(imageDeserializers)
            {
                targetCtfDeserializer
            };

            MinibatchSourceConfig config = new MinibatchSourceConfig(deserializerList)
            {
                MaxSweeps = (uint)maxEpochsOrSweeps,
            };
            
            return CNTKLib.CreateCompositeMinibatchSource(config);
        }

        static void AddGrayScaleTrue(CNTKDictionary deserializer)
        {
            deserializer.Add("grayscale", new DictionaryValue(true));
        }

        static CNTKDictionary CreateCtfDeserializer(string ctfFilePath, int outputShape)
        {
            var ctfStreamConfigurationVector = new StreamConfigurationVector();
            ctfStreamConfigurationVector.Add(new StreamConfiguration(TargetsName, outputShape, isSparse: false));

            var targetCtfDeserializer = CTFDeserializer(ctfFilePath, ctfStreamConfigurationVector);
            return targetCtfDeserializer;
        }
    }
}
