package opt.test;

import dist.*;
import opt.*;
import opt.example.*;
import opt.ga.*;
import shared.*;
import func.nn.backprop.*;

import java.util.*;
import java.io.*;
import java.text.*;

/**
 * Implementation of randomized hill climbing, simulated annealing, and genetic algorithm to
 * find optimal weights to a neural network that is classifying satellite images as showing 
 * diseased (1) or healthy (0) trees. 
 *
 * @author Hannah Lau
 * @version 1.0
 */
public class WiltSA {
    private static Instance[] instances = initializeInstances();

    private static int inputLayer = 5, hiddenLayer = 3, outputLayer = 1, trainingIterations = 100;
    private static BackPropagationNetworkFactory factory = new BackPropagationNetworkFactory();
    
    private static ErrorMeasure measure = new SumOfSquaresError();

    private static DataSet set = new DataSet(instances);

    private static BackPropagationNetwork networks[] = new BackPropagationNetwork[3];
    private static NeuralNetworkOptimizationProblem[] nnop = new NeuralNetworkOptimizationProblem[3];

    private static OptimizationAlgorithm[] oa = new OptimizationAlgorithm[3];
    private static String[] oaNames = {"RHC", "SA", "GA"};
    private static String results = "";

    private static DecimalFormat df = new DecimalFormat("0.000");

    public static void main(String[] args) {
        for(int i = 0; i < oa.length; i++) {
            networks[i] = factory.createClassificationNetwork(
                new int[] {inputLayer, hiddenLayer, outputLayer});
            nnop[i] = new NeuralNetworkOptimizationProblem(set, networks[i], measure);
        }

        oa[0] = new RandomizedHillClimbing(nnop[0]);
        oa[1] = new SimulatedAnnealing(1E11, 1, nnop[1]);
        oa[2] = new StandardGeneticAlgorithm(200, 100, 10, nnop[2]);

        for(int i = 1; i < 2; i++) {
            double start = System.nanoTime(), end, trainingTime, testingTime, correct = 0, incorrect = 0;
            train(oa[i], networks[i], oaNames[i]); //trainer.train();
            end = System.nanoTime();
            trainingTime = end - start;
            trainingTime /= Math.pow(10,9);

            Instance optimalInstance = oa[i].getOptimal();
            networks[i].setWeights(optimalInstance.getData());

            double predicted, actual;
            start = System.nanoTime();
            for(int j = 0; j < instances.length; j++) {
                networks[i].setInputValues(instances[j].getData());
                networks[i].run();

                predicted = Double.parseDouble(instances[j].getLabel().toString());
                actual = Double.parseDouble(networks[i].getOutputValues().toString());

                double trash = Math.abs(predicted - actual) < 0.5 ? correct++ : incorrect++;

            }
            end = System.nanoTime();
            testingTime = end - start;
            testingTime /= Math.pow(10,9);

	    // Print percent corrent, training time, and testing time separated by commas
	    results +=  df.format(correct/(correct+incorrect)*100) + "," + df.format(trainingTime) +
                        "," + df.format(testingTime) + ",";
        }

	// Format output to a single line that can be copied into a text doc for copying to CSV
	// InputSize, HiddenSize, OutputSize, Iterations, Temperature, Cooling, Percent Correct, Train time, Test time
	results = String.valueOf(inputLayer) + "," +
	    String.valueOf(hiddenLayer) + ","  +
	    String.valueOf(outputLayer) + "," +
	    String.valueOf(trainingIterations) + ","  +
	    "1E11" + "," +
	    "1" + "," +
	    results.substring(0, results.length() - 1);

	// Write output to CSV file
	System.out.println(results);
	try (Writer writer = new BufferedWriter(new FileWriter("/Users/Mike/Documents/OMSCS/ML/Projects/Randomized Optimization/data/weight_optimization_sa.csv", true))) {
	    writer.append("\n" + results);
	}
	catch (IOException e) {
            e.printStackTrace();
        }
	
    }

    private static void train(OptimizationAlgorithm oa, BackPropagationNetwork network, String oaName) {
        System.out.println("\nError results for " + oaName + "\n---------------------------");

        for(int i = 0; i < trainingIterations; i++) {
            oa.train();

            double error = 0;
            for(int j = 0; j < instances.length; j++) {
                network.setInputValues(instances[j].getData());
                network.run();

                Instance output = instances[j].getLabel(), example = new Instance(network.getOutputValues());
                example.setLabel(new Instance(Double.parseDouble(network.getOutputValues().toString())));
                error += measure.value(output, example);
            }

            System.out.println(df.format(error));
        }
    }

    private static Instance[] initializeInstances() {

        double[][][] attributes = new double[4339][][]; // 4339 instances

        try {
            BufferedReader br = new BufferedReader(new FileReader(new File("/Users/Mike/Documents/OMSCS/ML/ABAGAIL/src/opt/test/wilt_training.txt")));

            for(int i = 0; i < attributes.length; i++) {
                Scanner scan = new Scanner(br.readLine());
                scan.useDelimiter(",");

                attributes[i] = new double[2][];
                attributes[i][0] = new double[5]; // 5 attributes
                attributes[i][1] = new double[1];

                for(int j = 0; j < 5; j++)
                    attributes[i][0][j] = Double.parseDouble(scan.next());

                attributes[i][1][0] = Double.parseDouble(scan.next());
            }
        }
        catch(Exception e) {
            e.printStackTrace();
        }

        Instance[] instances = new Instance[attributes.length];

        for(int i = 0; i < instances.length; i++) {
            instances[i] = new Instance(attributes[i][0]);
            instances[i].setLabel(new Instance(attributes[i][1][0]));
        }

        return instances;
    }
}
