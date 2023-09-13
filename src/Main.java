package src;

public class Main {
    public static void main(String args[]) {

        // initialization of Controller
        Controller controller = new Controller();
        Thread controllerThread = new Thread(controller);
        controllerThread.start();
        try {
            controllerThread.join();
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }
}
