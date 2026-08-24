package src.api;

import java.io.IOException;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;

import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;

import src.models.Layer;
import src.models.Network;
import src.models.Neuron;

public class PredictionServer {

    private static final String HOST = "localhost";
    private static final int PORT = 8080;
    private final Network network;

    public PredictionServer(Network network) {
        this.network = network;
    }

    public void startAndWait() throws IOException, InterruptedException {
        HttpServer server = HttpServer.create(new InetSocketAddress(HOST, PORT), 0);
        server.createContext("/health", this::handleHealth);
        server.createContext("/predict", this::handlePredict);
        server.setExecutor(Executors.newSingleThreadExecutor());
        server.start();
        new CountDownLatch(1).await();
    }

    private void handleHealth(HttpExchange exchange) throws IOException {
        if (!"GET".equalsIgnoreCase(exchange.getRequestMethod())) {
            sendResponse(exchange, 405, "{\"error\":\"Only GET is supported\"}");
            return;
        }
        sendResponse(exchange, 200, "{\"status\":\"ok\"}");
    }

    private void handlePredict(HttpExchange exchange) throws IOException {
        if (!"POST".equalsIgnoreCase(exchange.getRequestMethod())) {
            sendResponse(exchange, 405, "{\"error\":\"Only POST is supported\"}");
            return;
        }
        try {
            String requestBody = new String(exchange.getRequestBody().readAllBytes(), StandardCharsets.UTF_8);
            double[] inputs = parseInputs(requestBody);
            int expectedInputs = network.getLayers()[0].getNeurons().length;
            if (inputs.length != expectedInputs) {
                sendResponse(exchange, 400, "{\"error\":\"Expected " + expectedInputs + " input values\"}");
                return;
            }
            network.forward(inputs);
            sendResponse(exchange, 200, createOutputResponse());
        } catch (IllegalArgumentException e) {
            sendResponse(exchange, 400, "{\"error\":\"" + e.getMessage() + "\"}");
        } catch (Exception e) {
            sendResponse(exchange, 500, "{\"error\":\"Prediction failed\"}");
        }
    }

    private double[] parseInputs(String requestBody) {
        int keyStart = requestBody.indexOf("\"inputs\"");
        if (keyStart < 0) throw new IllegalArgumentException("Missing inputs array");
        int arrayStart = requestBody.indexOf('[', keyStart);
        int arrayEnd = requestBody.indexOf(']', arrayStart);
        if (arrayStart < 0 || arrayEnd < 0) throw new IllegalArgumentException("Invalid inputs array");
        String valuesText = requestBody.substring(arrayStart + 1, arrayEnd).trim();
        if (valuesText.isEmpty()) return new double[0];
        String[] values = valuesText.split(",");
        double[] inputs = new double[values.length];
        for (int i = 0; i < values.length; i++) {
            try {
                inputs[i] = Double.parseDouble(values[i].trim());
            } catch (NumberFormatException e) {
                throw new IllegalArgumentException("Inputs must contain only numbers");
            }
        }
        return inputs;
    }

    private String createOutputResponse() {
        Layer outputLayer = network.getLayers()[network.getLayers().length - 1];
        StringBuilder response = new StringBuilder("{\"outputs\":[");
        Neuron[] neurons = outputLayer.getNeurons();
        for (int i = 0; i < neurons.length; i++) {
            if (i > 0) response.append(',');
            response.append(neurons[i].getActivation());
        }
        return response.append("]}").toString();
    }

    private void sendResponse(HttpExchange exchange, int statusCode, String body) throws IOException {
        byte[] response = body.getBytes(StandardCharsets.UTF_8);
        exchange.getResponseHeaders().set("Content-Type", "application/json");
        exchange.sendResponseHeaders(statusCode, response.length);
        try (OutputStream output = exchange.getResponseBody()) {
            output.write(response);
        }
    }
}
