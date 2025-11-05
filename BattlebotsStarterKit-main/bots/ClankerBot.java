package bots;

import arena.BattleBotArena;
import arena.BotInfo;
import arena.Bullet;
import java.awt.Graphics;
import java.awt.Image;
import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;

/**
 * PPOBot queries a Python web server (running a Stable Baselines 3 model)
 * to determine its next move
 *
 * it translates the Java game state into the JSON format expected
 * by the server and parses the returned action
 */

public class ClankerBot extends Bot {

    private static final String SERVER_URL = "http://127.0.0.1:5000";
    
    private static final int CONNECT_TIMEOUT_MS = 25;
    private static final int READ_TIMEOUT_MS = 25;


    private boolean isFirstMove = true;

    
    private Image currentImage;
    private Image[] botImages = new Image[4]; 

    
    private String botName = null;

    
    @Override
    public void newRound() {
        
        this.isFirstMove = true;
        
        
        if (botImages[0] != null) {
            this.currentImage = botImages[0];
        }
    }

    
    @Override
    public int getMove(BotInfo me, boolean shotOK, BotInfo[] liveBots, BotInfo[] deadBots, Bullet[] bullets) {
        
        try {
            
            String jsonPayload = buildJsonPayload(me, shotOK, liveBots, bullets);

            
            String endpoint = this.isFirstMove ? "/reset" : "/predict";

            
            int action = queryServer(jsonPayload, endpoint);

            
            if (this.isFirstMove) {
                this.isFirstMove = false;
            }
            
            
            updateImage(action);

            
            return action;

        } catch (Exception e) {
            
            
            System.err.println("PPOBot Error: " + e.getMessage());
            
            return BattleBotArena.STAY;
        }
    }

    
    private String buildJsonPayload(BotInfo me, boolean shotOK, BotInfo[] liveBots, Bullet[] bullets) {
        StringBuilder json = new StringBuilder();
        json.append("{");

        
        json.append("\"agent_pos\": [").append(me.getX()).append(", ").append(me.getY()).append("],");

        
        
        
        
        
        
        int ammoToSend = shotOK ? BattleBotArena.NUM_BULLETS : 0;
        json.append("\"agent_ammo\": ").append(ammoToSend).append(",");

        
        json.append("\"other_bots\": [");
        boolean firstBot = true;
        for (BotInfo bot : liveBots) {
            if (!firstBot) {
                json.append(",");
            }
            
            
            
            int botAmmo = BattleBotArena.NUM_BULLETS;
            json.append("[").append(bot.getX()).append(", ").append(bot.getY()).append(", ").append(botAmmo).append("]");
            firstBot = false;
        }
        json.append("],");

        
        json.append("\"bullets\": [");
        boolean firstBullet = true;
        for (Bullet b : bullets) {
            if (b != null) {
                if (!firstBullet) {
                    json.append(",");
                }
                json.append("[").append(b.getX()).append(", ").append(b.getY()).append("]");
                firstBullet = false;
            }
        }
        json.append("]");

        json.append("}");
        return json.toString();
    }

    
    private int queryServer(String jsonPayload, String endpoint) throws Exception {
        URL url = new URL(SERVER_URL + endpoint);
        HttpURLConnection conn = (HttpURLConnection) url.openConnection();

        
        conn.setRequestMethod("POST");
        conn.setRequestProperty("Content-Type", "application/json; utf-8");
        conn.setRequestProperty("Accept", "application/json");
        conn.setDoOutput(true);
        conn.setConnectTimeout(CONNECT_TIMEOUT_MS);
        conn.setReadTimeout(READ_TIMEOUT_MS);

        
        try (OutputStream os = conn.getOutputStream()) {
            byte[] input = jsonPayload.getBytes(StandardCharsets.UTF_8);
            os.write(input, 0, input.length);
        }

        
        StringBuilder response = new StringBuilder();
        try (BufferedReader br = new BufferedReader(new InputStreamReader(conn.getInputStream(), StandardCharsets.UTF_8))) {
            String responseLine;
            while ((responseLine = br.readLine()) != null) {
                response.append(responseLine.trim());
            }
        }

        
        String jsonResponse = response.toString();
        int actionIndex = jsonResponse.indexOf("\"action\"");
        if (actionIndex == -1) {
            throw new Exception("Invalid response from server: " + jsonResponse);
        }
        
        
        String sub = jsonResponse.substring(actionIndex + 8); 
        sub = sub.replaceAll("[^\\d]", ""); 
        
        if (sub.isEmpty()) {
             throw new Exception("Could not parse action from server response: " + jsonResponse);
        }

        return Integer.parseInt(sub);
    }
    
    
    private void updateImage(int action) {
        switch (action) {
            case BattleBotArena.UP:
            case BattleBotArena.FIREUP:
                this.currentImage = this.botImages[0];
                break;
            case BattleBotArena.DOWN:
            case BattleBotArena.FIREDOWN:
                this.currentImage = this.botImages[1];
                break;
            case BattleBotArena.LEFT:
            case BattleBotArena.FIRELEFT:
                this.currentImage = this.botImages[2];
                break;
            case BattleBotArena.RIGHT:
            case BattleBotArena.FIRERIGHT:
                this.currentImage = this.botImages[3];
                break;
            
        }
    }

    

    @Override
    public void draw(Graphics g, int x, int y) {
        if (this.currentImage != null) {
            g.drawImage(this.currentImage, x, y, Bot.RADIUS * 2, Bot.RADIUS * 2, null);
        } else {
            
            g.setColor(java.awt.Color.CYAN);
            g.fillOval(x, y, Bot.RADIUS * 2, Bot.RADIUS * 2);
        }
    }

    @Override
    public String getName() {
        if (this.botName == null) {
            this.botName = "PPOBot" + (botNumber < 10 ? "0" : "") + botNumber;
        }
        return this.botName;
    }

    @Override
    public String getTeamName() {
        return "PPO"; 
    }

    @Override
    public String[] imageNames() {
        
        return new String[]{"roomba_up.png", "roomba_down.png", "roomba_left.png", "roomba_right.png"};
    }

    @Override
    public void loadedImages(Image[] images) {
        if (images != null && images.length == 4) {
            this.botImages[0] = images[0]; 
            this.botImages[1] = images[1]; 
            this.botImages[2] = images[2]; 
            this.botImages[3] = images[3]; 
            this.currentImage = this.botImages[0]; 
        }
    }

    @Override
    public String outgoingMessage() {
        return null; 
    }

    @Override
    public void incomingMessage(int botNum, String msg) {
        
    }
}