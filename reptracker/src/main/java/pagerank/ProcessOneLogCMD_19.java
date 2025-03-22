package pagerank;

import org.jgrapht.graph.DirectedPseudograph;
import org.json.simple.JSONArray;
import org.json.simple.JSONObject;

import java.io.*;
import java.sql.Timestamp;
import java.util.*;
import java.util.stream.Collectors;

/**
 * Created by fang on 4/6/18.
 */
@SuppressWarnings("Duplicates")
public class ProcessOneLogCMD_19 {
    static OutputStream os = null;
    public static int topStarts = 3;     //parameter for choosing top N starts for forward analysis

    public static void process_backward(String resultDir, String suffix, double threshold, boolean trackOrigin, String logfile, String[] IP, String detection,String[] highRP,String[] midRP,String[] lowRP, String filename,double detectionSize, Set<String> seedSources,String[] criticalEdges, String mode, JSONObject jsonLog){
        OutputStream weightfile = null;
        try{
            os = new FileOutputStream(resultDir+filename+suffix+"_stats");
            GetGraph getGraph = new GetGraph(logfile, IP);
            long startTime = System.currentTimeMillis();
            getGraph.GenerateGraph();
            DirectedPseudograph<EntityNode, EventEdge> orignal = getGraph.getJg();
            System.out.println("Original vertex number:" + orignal.vertexSet().size() + " edge number : " + orignal.edgeSet().size());
            os.write(("Original vertex number:" + orignal.vertexSet().size() + " edge number : " + orignal.edgeSet().size()+"\n").getBytes());
            long endTime = System.currentTimeMillis();
            double timeCost = getTimeCost(startTime, endTime);
            System.out.println("Build Original Graph time cost is: "+ timeCost);
            os.write(("Build Original Graph time cost is: "+ timeCost+"\n").getBytes());
            jsonLog.put("origionVertexNumber", orignal.vertexSet().size());
            jsonLog.put("origionEdgeNumber", orignal.edgeSet().size());
            jsonLog.put("CostForOrigionGraph", timeCost);
            run_exp_backward(orignal, resultDir, suffix, threshold, trackOrigin, logfile, IP, detection,highRP,midRP,lowRP,filename, detectionSize,seedSources,criticalEdges, mode, jsonLog);

        }catch (Exception e){
            e.printStackTrace();
        }finally {
            try{
//                weightfile.close();
            }catch (Exception e){
                e.printStackTrace();
            }
        }
    }
    //backtrack + backward propagate
    public static void run_exp_backward(DirectedPseudograph<EntityNode, EventEdge> orignal,String resultDir, String suffix, double threshold, boolean trackOrigin, String logfile, String[] IP, String detection,String[] highRP,String[] midRP,String[] lowRP, String filename,double detectionSize, Set<String> seedSources,String[] criticalEdges, String mode, JSONObject jsonlog){
        OutputStream weightfile = null;
        try{
            os = new FileOutputStream(resultDir+filename+suffix+"_stats");
            long start = System.currentTimeMillis();
            BackTrack backTrack = new BackTrack(orignal);
            backTrack.backTrackPOIEvent(detection);
            long end = System.currentTimeMillis();
            double timeCost = getTimeCost(start, end);
            System.out.println("BackTrack time cost is: "+timeCost);
            os.write(("BackTrack time cost is: "+timeCost+"\n").getBytes());
            System.out.println("After Backtrack vertex number is: "+ backTrack.afterBackTrack.vertexSet().size() + " edge number: " + backTrack.afterBackTrack.edgeSet().size());
            os.write(("After Backtrack vertex number is: "+ backTrack.afterBackTrack.vertexSet().size() + " edge number: " + backTrack.afterBackTrack.edgeSet().size()+"\n").getBytes());
            jsonlog.put("BackTrackVertexNumber", backTrack.afterBackTrack.vertexSet().size());
            jsonlog.put("BackTrackEdgeNumber", backTrack.afterBackTrack.edgeSet().size());
            jsonlog.put("BackTrackTimeCost", timeCost);
            IterateGraph out = new IterateGraph(backTrack.afterBackTrack);
            out.exportGraph(resultDir+"BackTrack_"+filename+suffix);
//            backTrack.exportGraph("backTrack");

//            // If we run cpr1 and cpr2 together, the cpr1 result will effect cpr2, need more investigation
//            CausalityPreserve CPR1 = new CausalityPreserve(backTrack.afterBackTrack);
//            CPR1.CPR(1);
//            System.out.println("After CPR1 vertex number is: "+ CPR1.afterMerge.vertexSet().size() + " edge number: " + CPR1.afterMerge.edgeSet().size());
//            os.write(("After CPR1 vertex number is: "+ CPR1.afterMerge.vertexSet().size() + " edge number: " + CPR1.afterMerge.edgeSet().size()+"\n").getBytes());
            CausalityPreserve CPR = new CausalityPreserve(backTrack.afterBackTrack);
            //CPR.CPR(2);
            start = System.currentTimeMillis();
            double timeWindow = 10.0;
            CPR.mergeEdgeFallInTheRange2(timeWindow);
            System.out.println("The size of time window is :" + timeWindow+"(s)");
            end = System.currentTimeMillis();
            timeCost = getTimeCost(start, end);
            System.out.println("Edge Merge cost is: "+timeCost);
            os.write(("Edge Merge cost is: "+timeCost+"\n").getBytes());
            System.out.println("After CPR vertex number is: "+ CPR.afterMerge.vertexSet().size() + " edge number: " + CPR.afterMerge.edgeSet().size());
            os.write(("After CPR vertex number is: "+ CPR.afterMerge.vertexSet().size() + " edge number: " + CPR.afterMerge.edgeSet().size()+"\n").getBytes());
            ProcessOneLogCMD_19.putToJsonLog(jsonlog, "CPRVertexNumber", String.valueOf(CPR.afterMerge.vertexSet().size()));
            ProcessOneLogCMD_19.putToJsonLog(jsonlog, "CPREdgeNumber", String.valueOf(CPR.afterMerge.edgeSet().size()));
            ProcessOneLogCMD_19.putToJsonLog(jsonlog, "CPRTimeCost", String.valueOf(timeCost));
            out = new IterateGraph(CPR.afterMerge);
            out.exportGraph(resultDir+"AfterCPR_"+filename+suffix);


            BackwardPropagate_pf infer = new BackwardPropagate_pf(CPR.afterMerge);


            infer.setDetectionSize(detectionSize);
            infer.setSeedSources(seedSources);   // set structure weight for source
            start = System.currentTimeMillis();
            switch (mode){
                case "nonml":
                    infer.calculateWeights();
                    break;
                case "clusterall":
                    infer.calculateWeights_ML_dec(true,1, resultDir);
                    break;
                case "nonoutlier":
                    infer.calculateWeights_ML_dec(true,2, resultDir);
                    break;
                case "clusterlocal_dec":
                    infer.calculateWeights_ML_dec(true,3, resultDir);
                    break;
                case "clusterlocal":
                    infer.calculateWeights_ML_dec(true, 3, resultDir);
                    break;
                case "localtime":
                    infer.calculateWeights_Individual(true,"timeWeight");
                    break;
                case "localamount":
                    infer.calculateWeights_Individual(true,"amountWeight");
                    break;
                case "localstruct":
                    infer.calculateWeights_Individual(true,"structureWeight");
                    break;
                case "fanout":
                    //InferenceReputation fanoutCalculate = new InferenceReputation(CPR.afterMerge);
                    infer.calculateWeights_Fanout(true, resultDir);
                    break;
                case "nonmlrandom":
                    infer.calculateWeightsRandom();
                    break;
                default:
                    System.out.println("Unknown mode: "+mode);
            }
            end = System.currentTimeMillis();
            timeCost = getTimeCost(start, end);
            String timeCostInfo = String.format("Weight Calculation (%s) time cost is: ", mode)+timeCost+"\n";
            System.out.println(timeCostInfo);
            os.write(timeCostInfo.getBytes());
            ProcessOneLogCMD_19.putToJsonLog(jsonlog, "WeightCalculationTimeCost", String.valueOf(timeCost));

//            infer.calculateWeights_ML(true);
//            infer.calculateWeights_Individual(true, "timeWeight");
//            infer.calculateWeights();
            //System.out.println("OTSU: "+infer.OTSUThreshold());


            List<String> skipmode = new ArrayList<>();
            skipmode.add("fanout");
            if(!skipmode.contains(mode)) {
                infer.initialReputation(highRP, lowRP);
                start = System.currentTimeMillis();
                infer.PageRankIterationBackward(highRP, midRP, lowRP, detection);
                end = System.currentTimeMillis();
                timeCost = getTimeCost(start, end);
                timeCostInfo = "Propagation time cost is: " + timeCost + "\n";
                System.out.println(timeCostInfo);
                os.write(timeCostInfo.getBytes());
                ProcessOneLogCMD_19.putToJsonLog(jsonlog, "PropagationTimeCost", String.valueOf(timeCost));

                infer.exportGraph(resultDir + "Weight_" + filename + suffix);
                List<List<String>> forwardStarts = infer.getForwardStarts();
                List<String> nodesignatures = infer.graph.vertexSet().stream().map(v -> v.getSignature()).collect(Collectors.toList());
                List<String> randomStarts = IterateGraph.getRandomStarts(nodesignatures, 3);  // not based on category
                List<List<String>> randomStartsCategory = IterateGraph.randomPickEntryStartsBasedOnCategory(forwardStarts);
                Map<String, Double> nodeReputation = IterateGraph.getNodeReputation(infer.graph);
                IterateGraph.outputTopStarts(resultDir, forwardStarts, nodeReputation);
                boolean outputFilterGraph = true;
                ProcessOneLogCMD_19.filter_graph_by_forward_category(forwardStarts, orignal, "sysrep", resultDir,
                        filename, suffix, "1", 3, infer, outputFilterGraph);
                outputFilterGraph = true;

                for (int i = 0; i < 20; i++) {
                    randomStartsCategory = IterateGraph.randomPickEntryStartsBasedOnCategory(forwardStarts);
                    randomStarts = IterateGraph.getRandomStarts(nodesignatures, 3);
                    ProcessOneLogCMD_19.filter_graph_by_forward_category(randomStartsCategory, orignal, "randomCategory",
                            resultDir, filename, suffix, String.valueOf(i), 3, infer, outputFilterGraph);
                    ProcessOneLogCMD_19.filter_graph_by_forward(randomStarts, orignal, "random", resultDir, filename,
                            suffix, String.valueOf(i), infer, outputFilterGraph);
                }


//            DirectedPseudograph<EntityNode, EventEdge> backresFilteredByForwad = infer.
//                    combineBackwardAndForwardForGivenStarts(forwardStarts, orignal);
//            out = new IterateGraph(backresFilteredByForwad);
//            out.exportGraph(resultDir+"BackTrack_Filtered_By_Forward"+filename+suffix);
                List<String> entryPoints = IterateGraph.getCandidateEntryPoint(infer.graph);
                JSONObject entryJson = new JSONObject();
                entryJson.put("EntryPointsNumber", entryPoints.size());
                ProcessOneLogCMD_19.putToJsonLog(jsonlog, "EntryPointsNumber", String.valueOf(entryPoints.size()));
                JSONArray ponintsJson = new JSONArray();
                entryPoints.stream().forEach(s -> ponintsJson.add(s));
                entryJson.put("EntryPoints", ponintsJson);
                File entryPointsJsonFile = new File(resultDir + filename + suffix + "_entry_points.json");
                FileWriter jsonWriter = new FileWriter(entryPointsJsonFile);
                jsonWriter.write(entryJson.toJSONString());
                jsonWriter.close();
            }
        }catch (Exception e){
            e.printStackTrace();
        }finally {
            try{
                os.close();
            }catch (Exception e){
                e.printStackTrace();
            }
        }
    }



    public static double getTimeCost(long start, long end){
        return (end-start)*1.0/1000.0;
    }

    public static Timestamp getTimeStamp(){
        Calendar calendar = Calendar.getInstance();
        Timestamp currentTimestamp = new java.sql.Timestamp(calendar.getTime().getTime());
        return currentTimestamp;
    }

    public static void putToJsonLog(JSONObject jsonLog, String key, String value){
        jsonLog.put(key, value);
    }

    public static void filter_graph_by_forward_category(List<List<String>> starts, DirectedPseudograph<EntityNode, EventEdge> orignal,
                                               String method, String resultDir, String filename, String suffix, String time, int
                                                        startLimitForEachCategory, BackwardPropagate_pf infer, boolean outputGraph
                                                            ){
        try{
            File resFolderForFilter = new File(resultDir+"/"+method);
            if (!resFolderForFilter.exists()){
                resFolderForFilter.mkdir();
            }
            File folderForFtime = new File(resFolderForFilter.getAbsolutePath()+"/"+time);
            if(!folderForFtime.exists()){
                folderForFtime.mkdir();
            }
            File recordStarts = new File(folderForFtime.getAbsolutePath()+"/"+"forward_starts_"+filename+"_"+time+"_"+method+".txt");
            FileWriter fileWriter = new FileWriter(recordStarts);
            PrintWriter printWriter = new PrintWriter(fileWriter);
            printWriter.println("Entry Points for forward:");
            int startsNum = 0;
            for(List<String> start : starts){
                for(int r = 0; r < startLimitForEachCategory && r <start.size(); r++){
                    printWriter.println(start.get(r));
                    DirectedPseudograph<EntityNode, EventEdge> backresFilteredByForwad = infer.
                            combineBackwardAndForwardForGivenStart(start.get(r), orignal);
                    IterateGraph out = new IterateGraph(backresFilteredByForwad);
                    if(outputGraph) {
                        out.exportGraph(folderForFtime.getAbsolutePath()+"/" + "filtered_by_forward_" + String.valueOf(startsNum) + "_" + filename + "_" + method + suffix);
                    }
                    startsNum++;
                }
            }

            printWriter.close();
        }catch (Exception e){
            e.printStackTrace();
        }
    }

    public static void filter_graph_by_forward(List<String> start,DirectedPseudograph<EntityNode, EventEdge> orignal,
                                               String method, String resultDir, String filename, String suffix, String time, BackwardPropagate_pf infer, boolean output){
        try {
            File resFolderForFilter = new File(resultDir+"/"+method);
            if (!resFolderForFilter.exists()){
                resFolderForFilter.mkdir();
            }
            File folderForFtime = new File(resFolderForFilter.getAbsolutePath()+"/"+time);
            if(!folderForFtime.exists()){
                folderForFtime.mkdir();
            }
            File recordStarts = new File(folderForFtime.getAbsolutePath()+"/"+"forward_starts_"+filename+"_"+time+"_"+method+".txt");
            FileWriter fileWriter = new FileWriter(recordStarts);
            PrintWriter printWriter = new PrintWriter(fileWriter);
            printWriter.println("Entry Points for forward:");
            int startsNum = 0;
            for(String s : start){
                printWriter.println(s);
                DirectedPseudograph<EntityNode, EventEdge> backresFilteredByForwad = infer.
                        combineBackwardAndForwardForGivenStart(s, orignal);
                System.out.println("Size of randome start: "+ backresFilteredByForwad.vertexSet().size());
                IterateGraph out = new IterateGraph(backresFilteredByForwad);
                if(output) {
                    out.exportGraph( folderForFtime.getAbsolutePath()+"/"+ "filtered_by_forward_" + String.valueOf(startsNum) + "_" + filename + "_" + method + suffix);
                }
                startsNum++;
            }
            printWriter.close();
        }catch (Exception e){
            e.printStackTrace();
        }

    }
//    public static void filter_graph_by_forward_category(List<List<String>> starts, DirectedPseudograph<EntityNode, EventEdge> orignal,
//                                                        String method, String resultDir, String filename, String suffix, String time, int
//                                                                startLimitForEachCategory, BackwardPropagate_pf infer){
//        try{
//
//            File recordStarts = new File(resultDir+"forward_starts_"+filename+"_"+time+"_"+method+".txt");
//            FileWriter fileWriter = new FileWriter(recordStarts);
//            PrintWriter printWriter = new PrintWriter(fileWriter);
//            printWriter.println("Entry Points for forward:");
//            int startsNum = 0;
//            for(List<String> start : starts){
//                for(int r = 0; r < startLimitForEachCategory && r <start.size(); r++){
//                    printWriter.println(start.get(r));
//                    DirectedPseudograph<EntityNode, EventEdge> backresFilteredByForwad = infer.
//                            combineBackwardAndForwardForGivenStart(start.get(r), orignal);
//                    IterateGraph out = new IterateGraph(backresFilteredByForwad);
//                    out.exportGraph(resultDir+"filtered_by_forward_"+String.valueOf(startsNum)+"_"+filename+"_"+method+suffix);
//                    startsNum++;
//                }
//            }
//            printWriter.close();
//        }catch (Exception e){
//            e.printStackTrace();
//        }
//    }
//    public static void filter_graph_by_forward(List<String> start,DirectedPseudograph<EntityNode, EventEdge> orignal,
//                                               String method, String resultDir, String filename, String suffix, String time, int
//                                                       startLimitForEachCategory, BackwardPropagate_pf infer){
//        try {
//            File recordStarts = new File(resultDir + "/" + "forward_starts_" + filename + "_" + time + "_" + method + ".txt");
//            FileWriter fileWriter = new FileWriter(recordStarts);
//            PrintWriter printWriter = new PrintWriter(fileWriter);
//            printWriter.println("Entry Points for forward:");
//            int startsNum = 0;
//            for(String s : start){
//                printWriter.println(s);
//                DirectedPseudograph<EntityNode, EventEdge> backresFilteredByForwad = infer.
//                        combineBackwardAndForwardForGivenStart(s, orignal);
//                IterateGraph out = new IterateGraph(backresFilteredByForwad);
//                out.exportGraph(resultDir+"filtered_by_forward_"+String.valueOf(startsNum)+"_"+filename+"_"+method+suffix);
//                startsNum++;
//            }
//            printWriter.close();
//        }catch (Exception e){
//            e.printStackTrace();
//        }
//    }


    public static double[] getMissingRateAndRedundantRate(DirectedPseudograph<EntityNode, EventEdge>graph, String[] criticalEdges){
        double missingNum = 0.0;
        Set<String> edges = new HashSet<>();
        for(EventEdge edge: graph.edgeSet()){
            edges.add(IterateGraph.convertEdgeToString(edge));
        }
        for(String s : criticalEdges){
            String[] srcAndTarget = s.split(",");
            String curEdge = srcAndTarget[0]+" -> "+srcAndTarget[1];
            if(!edges.contains(curEdge)){
                missingNum += 1;
            }
        }

        Set<String> stringRepOfCirticalEdges = convertCriticalEdge(criticalEdges);
        double redundantNum = 0.0;
        for(String e:edges){
            if(!stringRepOfCirticalEdges.contains(e)){
                redundantNum += 1;
            }
        }
        double[] res = new double[2];
        res[0] = missingNum/criticalEdges.length;
        res[1] = redundantNum/graph.edgeSet().size();
        return res;
    }

    public static Set<String> convertCriticalEdge(String[] criticalEdges){
        Set<String> res = new HashSet<>();
        for(String s : criticalEdges){
            String[] srcAndTarget = s.split(",");
            String curEdge = srcAndTarget[0]+" -> "+srcAndTarget[1];
            res.add(curEdge);
        }
        return res;
    }
}
