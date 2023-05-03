package src;
public class Main {
    public static void main(String args[]) {
        Controller controller = new Controller();
        Thread controllerThread = new Thread(controller);
        controllerThread.start();
    }
}
