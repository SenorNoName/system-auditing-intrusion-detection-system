package pagerank;

import org.jgrapht.graph.DirectedPseudograph;
import org.json.simple.JSONObject;

import java.io.*;
import java.sql.Timestamp;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.*;
import java.util.logging.*;
import java.util.logging.Formatter;
/*
Specify the path to log files and result directories. Either a list of log file names or a FileNameFormatter is acceptable.
The directory of log must also contains configuration files with the same name as the corresponding log but ends with ".property".
 */
public class ExperimentRunnerCmd {
    public static String mode;
    public static String do_split;
    DirectedPseudograph<EntityNode, EventEdge> graphFromLog;

    public static void main(String[] args){
        for(String arg : args)
            System.out.println(arg);

        if(args.length!=4){
            System.err.println("Usage: ExperimentRunnerCmd <log path> <result path> <log names,...> <mode>");
            System.exit(-1);
        }
        try{
            String logPath = args[0];
            String resPath = args[1];
            String[] logs = args[2].split(";");
            mode = args[3];
            //do_split = args[4];

            ExperimentRunnerCmd er = new ExperimentRunnerCmd(logPath, resPath, logs);

//            ExperimentRunner er = new ExperimentRunner(
//                    "input/attacks_fine",
//                    "results/exp",
//                    new String[] {"password_crack_step5.txt"});
//            er.run();
            er.run2();
        }catch (Exception e){
            e.printStackTrace();
        }

    }

    String PathToLogs;
    String PathToRes;
    FilenameFilter logNameFilter;

    public ExperimentRunnerCmd(String pathToLogs, String pathToRes, FilenameFilter logNameFilter){
        PathToLogs = pathToLogs;
        PathToRes = pathToRes;
        this.logNameFilter = logNameFilter;
    }

    public ExperimentRunnerCmd(String pathTologs, String pathToRes, String[] lognames){
        PathToLogs = pathTologs;
        PathToRes = pathToRes;
        Set<String> logset = new HashSet<>();
        for(String s: lognames) logset.add(s);
        logNameFilter = (dir,name) -> logset.contains(name);
    }

    public ExperimentRunnerCmd(String pathToLogs, String pathToRes, FilenameFilter logNameFilter, String[] exclusion){
        PathToLogs = pathToLogs;
        PathToRes = pathToRes;
        Set<String> exclusionSet = new HashSet<>();
        for(String s: exclusion) exclusionSet.add(s);
        this.logNameFilter = (dir, name) -> logNameFilter.accept(dir,name) && !exclusionSet.contains(name);
    }

    public void run() throws FileNotFoundException{

        File resDir = makeResDir(PathToRes);
        File[] logs = getLogs(PathToLogs);

        List<Experiment> experiments = new ArrayList<>();
        PrintStream logStream;
        try{
            for(File log : logs){
                File[] propertyFiles = log.getParentFile().listFiles((dir,name)->
                        (name.split("\\.")[0].equals(log.getName().split("\\.")[0]) || name.startsWith(log.getName().split("\\.")[0]+"-")) &&name.endsWith(".property"));
                for(File propertyFile : propertyFiles){
                    experiments.add(new Experiment(log,propertyFile));
                }
            }

            for(Experiment e : experiments){
                File oneRes = new File(resDir+"/"+e.configFile.getParentFile().getName()+"-"+e.configFile.getName().split("\\.")[0]);
                if(!oneRes.exists())
                    oneRes.mkdir();

                File logFile = new File(oneRes.getAbsolutePath()+"/"+e.log.getName().split("\\.")[0]+".log");
                logStream = new PrintStream(new FileOutputStream(logFile,true));
                logging(e,logFile);
                LogStream ls = new LogStream(System.out,logStream);
                LogStream lse = new LogStream(System.err,logStream);
                System.setOut(ls);
                System.setErr(lse);

                ProcessOneLogCmd.process(oneRes.getAbsolutePath()+"/", "", e.threshold, e.trackOrigin, e.log.getAbsolutePath(),MetaConfig.localIP,e.POI,e.highRP,e.midRP,e.lowRP,
                        e.log.getName().split("\\.")[0],e.detectionSize,e.getInitial(),e.criticalEdges,mode,do_split);
                logStream.close();
            }
        }catch (IOException e){
            e.printStackTrace();
        }
    }
    //Fang: for benign cases: avoid to parse log several times
    public void run2() throws FileNotFoundException{
        File resDir = makeResDir(PathToRes);
        File[] logs = getLogs(PathToLogs);
        File log = logs[0];
        List<Experiment> experiments = new ArrayList<>();
        List<Experiment> experiments_backward = new ArrayList<>();
        PrintStream logStream;
        try(FileOutputStream fo = new FileOutputStream(this.PathToRes+"/build_time.log")){
            long start = System.currentTimeMillis();
            GetGraph generator = new GetGraph(logs[0].getPath(), MetaConfig.localIP);
            long end = System.currentTimeMillis();
            fo.write(("parsing time:"+(end-start)/1000.0).getBytes());
            start = System.currentTimeMillis();
            generator.GenerateGraph();
            end = System.currentTimeMillis();
            fo.write(("building time:"+(end-start)/1000.0).getBytes());
            graphFromLog = generator.getJg();
        }catch(IOException e){}

        try{
            File[] propertyFiles = log.getParentFile().listFiles((dir,name)->
                    (name.split("\\.")[0].equals(log.getName().split("\\.")[0]) || name.startsWith(log.getName().split("\\.")[0]+":")) &&name.endsWith(".property"));
            for(File propertyFile:propertyFiles){
//                Experiment exp = new Experiment(logs[0],propertyFile);
//                experiments.add(exp);
                String propertyName = propertyFile.getName();
                if(propertyName.indexOf("backward")!=-1){
                    experiments_backward.add(new Experiment(log, propertyFile));
                }
            }

            for(Experiment e: experiments_backward){
//                File oneRes = new File(resDir+"/"+e.configFile.getParentFile().getName()+"-"+e.configFile.getName().split("\\.")[0]);
//                if(!oneRes.exists())
//                    oneRes.mkdir();
//
//                File logFile = new File(oneRes.getAbsolutePath()+"/"+e.log.getName().split("\\.")[0]+".log");
//                logStream = new PrintStream(new FileOutputStream(logFile,true));
//                logging(e,logFile);
//                LogStream ls = new LogStream(System.out,logStream);
//                LogStream lse = new LogStream(System.err,logStream);
//                System.setOut(ls);
//                System.setErr(lse);
//
//                ProcessOneLogCMD_19.run_exp_backward(graphFromLog,oneRes.getAbsolutePath()+"/", "", e.threshold, e.trackOrigin, e.log.getAbsolutePath(),MetaConfig.localIP,e.POI,e.highRP,e.midRP,e.lowRP,
//                        e.log.getName().split("\\.")[0],e.detectionSize,e.getInitial(),e.criticalEdges,mode,do_split);
//                logStream.close();
                File oneRes = new File(resDir + "/" + e.configFile.getParentFile().getName() + "-" + e.configFile.getName().split("\\.")[0]);
                if (!oneRes.exists())
                    oneRes.mkdir();

                File logFile = new File(oneRes.getAbsolutePath() + "/" + e.log.getName().split("\\.")[0] + ".log");
                logStream = new PrintStream(new FileOutputStream(logFile, true));
                logging(e, logFile);
                LogStream ls = new LogStream(System.out,logStream);
                LogStream lse = new LogStream(System.err,logStream);
                System.setOut(ls);
                System.setErr(lse);
                JSONObject jsonLog = new JSONObject();
                jsonLog.put("Case", e.log.getName());
                jsonLog.put("Mode", mode);
                ProcessOneLogCMD_19.run_exp_backward(graphFromLog,oneRes.getAbsolutePath() + "/", "", e.threshold, e.trackOrigin, e.log.getAbsolutePath(), MetaConfig.localIP, e.POI, e.highRP, e.midRP, e.lowRP, e.log.getName().split("\\.")[0], e.detectionSize, e.getInitial(), e.criticalEdges, mode, jsonLog);

//                ProcessOneLogCMD_19.process_backward(oneRes.getAbsolutePath() + "/", "", e.threshold, e.trackOrigin, e.log.getAbsolutePath(), MetaConfig.localIP, e.POI, e.highRP, e.midRP, e.lowRP, e.log.getName().split("\\.")[0], e.detectionSize, e.getInitial(), e.criticalEdges, "nonml");
                logStream.close();
                Timestamp currentTimestamp= ExperimentRunnerCMD_19.getTimeStamp();
                jsonLog.put("Timestamp", currentTimestamp.toString());
                File entryPointsJsonFile = new File(oneRes.getAbsolutePath()+"/"+e.configFile.getName().split("\\.")[0]+"_json_log.json");
                FileWriter jsonWriter = new FileWriter(entryPointsJsonFile);
                jsonWriter.write(jsonLog.toJSONString());
                jsonWriter.close();
            }
        }catch (IOException e){
            e.printStackTrace();
        }
    }

    private void logging(Experiment e, File logFile) throws IOException{
        Logger logger = Logger.getLogger(e.configFile.getName());
        FileHandler fileHandler = new FileHandler(logFile.getAbsolutePath());
        fileHandler.setLevel(Level.ALL);
        fileHandler.setFormatter(new LogFormatter());
        logger.addHandler(fileHandler);

        logger.info("#############Configurations#############");
        logger.info("Log File: "+e.log.getAbsolutePath());
        logger.info("Config File: "+e.configFile.getAbsolutePath());
        logger.info("POI: "+e.POI);
        logger.info("Local IP: "+formatArray(MetaConfig.localIP));
        logger.info("High RP: "+formatArray(e.highRP));
        logger.info("Low RP: "+formatArray(e.lowRP));
        logger.info("Threshold: "+String.valueOf(e.threshold));
        logger.info("Track Origin: "+String.valueOf(e.trackOrigin));
        logger.info("Detection size: "+String.valueOf(e.detectionSize));
        logger.info("Seeds: "+formatArray(e.getInitial().toArray(new String[e.getInitial().size()])));
        logger.info("##############Sysdig Parser#############");
        logger.info("P2P: "+formatArray(MetaConfig.ptopSystemCall));
        logger.info("P2F: "+formatArray(MetaConfig.ptofSystemCall));
        logger.info("F2P: "+formatArray(MetaConfig.ftopSystemCall));
        logger.info("P2N: "+formatArray(MetaConfig.ptonSystemCall));
        logger.info("N2P: "+formatArray(MetaConfig.ntopSystemCall));
        logger.info("########################################");
        logger.info("Mid RP: "+formatArray(e.midRP));
        logger.info("########################################");
    }

    private String formatArray(String[] array){
        return String.join(",",array);
    }

    private File makeResDir(String pathToRes){
        File resDir = new File(PathToRes);
        if(!resDir.exists() || !resDir.isDirectory())
            resDir.mkdir();
        else System.out.println("Result Directory "+pathToRes+" already exists, overwriting!");
        return resDir;
    }

    private File[] getLogs(String pathToLogs) throws FileNotFoundException{
        File logDir = new File(pathToLogs);
        if(!logDir.exists() || !logDir.isDirectory())
            throw new FileNotFoundException("Invalid directory: "+logDir);
        return logDir.listFiles(logNameFilter);
    }

    class LogFormatter extends Formatter {
        // Create a DateFormat to format the logger timestamp.
        private DateFormat df = new SimpleDateFormat("yyyy-MM-dd hh:mm:ss.SSS");

        public String format(LogRecord record) {
            StringBuilder builder = new StringBuilder();
            builder.append(df.format(new Date(record.getMillis()))).append(" - ");
            builder.append(formatMessage(record));
            builder.append("\n");
            return builder.toString();
        }
    }

    class LogStream extends PrintStream {
        PrintStream out;
        public LogStream(PrintStream out1, PrintStream out2) {
            super(out1);
            this.out = out2;
        }
        public void write(byte buf[], int off, int len) {
            try {
                super.write(buf, off, len);
                out.write(buf, off, len);
            } catch (Exception e) {
            }
        }
        public void flush() {
            super.flush();
            out.flush();
        }

    }
}
