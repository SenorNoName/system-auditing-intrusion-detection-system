package logparsers;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

public class Utils {
    public static Map<String, String> parseEntryWithoutID(String entry) {
        Map<String, String> res = new HashMap<>();
        String[] fields = entry.split(" ");
        res.put("raw",entry);
        res.put("timestamp",fields[0]);
        res.put("cpu",fields[1]);
        res.put("process",fields[2]);
        res.put("pid",fields[3].substring(1,fields[3].length()-1));
        res.put("direction", fields[4]);
        res.put("event", fields[5]);
        res.put("cwd", fields[6].substring(4));
        res.put("latency", fields[fields.length-1].substring(8));
        res.put("args", String.join(" ", Arrays.copyOfRange(fields,7,fields.length-2)));
        return res;
    }

    public static Map<String, String> parseEntry_without_exec(String entry) {
        Map<String, String> res = new HashMap<>();
        String[] fields = entry.split(" ");
        res.put("raw",entry);
        res.put("timestamp",fields[1]);
        res.put("cpu",fields[2]);
        res.put("process",fields[3]);
        res.put("pid",fields[4].substring(1,fields[4].length()-1));
        res.put("direction", fields[5]);
        res.put("event", fields[6]);
        res.put("cwd", fields[7].substring(4));
        res.put("latency", fields[fields.length-1].substring(8));
        res.put("args", String.join(" ", Arrays.copyOfRange(fields,7,fields.length-1)));
        return res;
    }

    public static Map<String, String> parseEntry(String entry) {
        Map<String, String> res = new HashMap<>();
        String[] fields = entry.split(" ");
        res.put("raw",entry);
        res.put("timestamp",fields[1]);
        res.put("cpu",fields[2]);
        res.put("process",fields[3]);
        res.put("pid",fields[4].substring(1,fields[4].length()-1));
        res.put("direction", fields[5]);
        res.put("event", fields[6]);
        res.put("cwd", fields[7].substring(4));
        if (fields[fields.length-1].startsWith("exepath")) {
            res.put("latency", fields[fields.length-2].substring(8));
            res.put("args", String.join(" ", Arrays.copyOfRange(fields,7,fields.length-2)));
        }else{
            res.put("latency", fields[fields.length-1].substring(8));
            res.put("args", String.join(" ", Arrays.copyOfRange(fields,7,fields.length-1)));
        }

        return res;
    }

    static long extractSize(String s) {
        if(s.contains("res=")) {
            String size = s.substring(s.indexOf("res=")+4).split(" ")[0];
            if (!size.startsWith("-"))
                return Long.parseLong(size);
        }
        return -1L;
    }

    static Map<String, String> extractFileandSocket(String s) {
        String path = null;
        String srcIP = null;
        String srcPort = null;
        String destIP = null;
        String destPort = null;
        String socket = null;

        Map<String, String> res = new HashMap<>();

        if(s.contains("fd=")) {
            String fd = s.substring(s.indexOf("fd=")).split(" ")[0];
            int index = 0;
            while (index < fd.length()) {
                if (fd.charAt(index) == '>') {
                    if (fd.charAt(index - 1) == 'f') {
                        path = fd.substring(index + 1, fd.length() - 1);
                        if(path.contains("("))
                            path = path.substring(path.indexOf("(")+1,path.length()-1);
                        break;
                    }
                    if (fd.charAt(index - 1) == 't' || fd.charAt(index - 1) == 'u') {
                        if (fd.charAt(index - 2) == '6' || fd.charAt(index - 2) == '4') {
                            socket = fd.substring(index + 1, fd.length() - 1);
                            if (!socket.equals("")) {
                                String[] portsAndIp = getIPandPorts(socket);             //0:src ip 1: src port 2:dest ip 3:dest port
                                srcIP = portsAndIp[0];
                                srcPort = portsAndIp[1];
                                destIP = portsAndIp[2];
                                destPort = portsAndIp[3];
                                break;
                            }
                        }
                    }
                    if(fd.charAt(index-1) == 'r' && fd.charAt(index-2) == '4'){
                        socket = fd.substring(index + 1, fd.length() - 1);
                        if (!socket.equals("")) {
                            String[] portsAndIp = getIPs(socket);             //0:src ip 1: src port 2:dest ip 3:dest port
                            srcIP = portsAndIp[0];
                            srcPort = "UNKNOWN";
                            destIP = portsAndIp[1];
                            destPort = "UNKNOWN";
                            break;
                        }
                    }
                }
                index++;
            }
        }

        if(path != null) res.put("path", path);
        if(srcIP != null) res.put("sip", srcIP);
        if(srcPort != null) res.put("sport", srcPort);
        if(destIP != null) res.put("dip", destIP);
        if(destPort != null) res.put("dport", destPort);
        if(socket != null && !socket.equals("")) res.put("socket", socket);
        return res;
    }

    static String extractProcessFile(String s) {
        String path = null;
        if(s.contains("filename=")){
            path = s.substring(s.indexOf("filename=")+9).split(" ")[0];
            if(path.contains("("))
                path = path.substring(path.indexOf("(")+1,path.length()-1);
        }
        return path;
    }

    private static String[] getIPandPorts(String str){
        String[] res = new String[4];
        String[] srcAndDest = str.split("->");
        if(srcAndDest.length<2){
            throw new ArrayIndexOutOfBoundsException("Can't parse socket!");
        }
        String[] src = srcAndDest[0].split(":");
        String[] dest = srcAndDest[1].split(":");
        res[0] = src[0];
        res[1] = src[1];
        res[2] = dest[0];
        res[3] = dest[1];
        if(res[3].equals("domamin")){
            res[3] = "53";
        }
        return res;
    }

    private static String[] getIPs(String str){
        String[] res = str.split("->");
        return res;
    }
}
