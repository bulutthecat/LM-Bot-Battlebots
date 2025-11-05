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
 * to determine its next move.
 *
 * It translates the Java game state into the JSON format expected
 * by the server and parses the returned action.
 */
public class ClankerBot extends Bot {

    /**
     * The URL of the Python Flask server.
     */
    private static final String SERVER_URL = "http://127.0.0.1:5000";
    
    /**
     * Timeouts in milliseconds.
     * The game runs at 30fps (33.3ms per frame). The server MUST respond
     * faster than this, or the bot will lag the entire game.
     * 25ms is a reasonable, aggressive timeout.
     */
    private static final int CONNECT_TIMEOUT_MS = 25;
    private static final int READ_TIMEOUT_MS = 25;

    /**
     * Flag to track if this is the first move of the round.
     * true = call /reset, false = call /predict
     */
    private boolean isFirstMove = true;

    /**
     * For drawing the bot's current image.
     */
    private Image currentImage;
    private Image[] botImages = new Image[4]; // 0=up, 1=down, 2=left, 3=right

    /**
     * Stores the bot's name.
     */
    private String botName = null;

    /**
     * Called at the start of each new round.
     * Resets the bot's state.
     */
    @Override
    public void newRound() {
        // This is the first move of the new round
        this.isFirstMove = true;
        
        // Default image
        if (botImages[0] != null) {
            this.currentImage = botImages[0];
        }
    }

    /**
     * This is the main logic method called by the arena every frame.
     *
     * @param me       Info about this bot.
     * @param shotOK   True if this bot is allowed to fire.
     * @param liveBots Array of info about all OTHER living bots.
     * @param deadBots Array of info about dead bots (obstacles).
     * @param bullets  Array of ALL active bullets on the field.
     * @return The integer action to take (e.g., BattleBotArena.UP).
     */
    @Override
    public int getMove(BotInfo me, boolean shotOK, BotInfo[] liveBots, BotInfo[] deadBots, Bullet[] bullets) {
        
        try {
            // 1. Build the JSON payload from the current game state
            String jsonPayload = buildJsonPayload(me, shotOK, liveBots, bullets);

            // 2. Determine which endpoint to call
            String endpoint = this.isFirstMove ? "/reset" : "/predict";

            // 3. Query the server
            int action = queryServer(jsonPayload, endpoint);

            // 4. If this was the first move, subsequent moves are /predict
            if (this.isFirstMove) {
                this.isFirstMove = false;
            }
            
            // 5. Update the bot's image based on the chosen action
            updateImage(action);

            // 6. Return the action to the arena
            return action;

        } catch (Exception e) {
            // Failsafe: If the server request fails (timeout, crash, etc.),
            // print the error and just stay still.
            System.err.println("PPOBot Error: " + e.getMessage());
            // e.printStackTrace(); // Uncomment for more detailed debugging
            return BattleBotArena.STAY;
        }
    }

    /**
     * Builds the JSON string payload expected by the Python server.
     */
    private String buildJsonPayload(BotInfo me, boolean shotOK, BotInfo[] liveBots, Bullet[] bullets) {
        StringBuilder json = new StringBuilder();
        json.append("{");

        // 1. Agent Position: [x, y]
        json.append("\"agent_pos\": [").append(me.getX()).append(", ").append(me.getY()).append("],");

        // 2. Agent Ammo: int
        // **CRITICAL COMPROMISE**: The Java framework doesn't provide the exact ammo
        // count, only 'shotOK'. We'll proxy this:
        // shotOK=true -> Send max ammo
        // shotOK=false -> Send 0 ammo
        // This gives the model a binary "can-fire/cannot-fire" signal.
        int ammoToSend = shotOK ? BattleBotArena.NUM_BULLETS : 0;
        json.append("\"agent_ammo\": ").append(ammoToSend).append(",");

        // 3. Other Bots: [[x, y, ammo], [x, y, ammo], ...]
        json.append("\"other_bots\": [");
        boolean firstBot = true;
        for (BotInfo bot : liveBots) {
            if (!firstBot) {
                json.append(",");
            }
            // **CRITICAL COMPROMISE**: We cannot know any other bot's ammo.
            // We must send a consistent lie. We'll send MAX_AMMO, assuming
            // all enemies are always "dangerous" (full ammo).
            int botAmmo = BattleBotArena.NUM_BULLETS;
            json.append("[").append(bot.getX()).append(", ").append(bot.getY()).append(", ").append(botAmmo).append("]");
            firstBot = false;
        }
        json.append("],");

        // 4. Bullets: [[x, y], [x, y], ...]
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

    /**
     * Sends the JSON payload to the server and parses the integer action
     * from the response.
     */
    private int queryServer(String jsonPayload, String endpoint) throws Exception {
        URL url = new URL(SERVER_URL + endpoint);
        HttpURLConnection conn = (HttpURLConnection) url.openConnection();

        // Set timeouts and request method
        conn.setRequestMethod("POST");
        conn.setRequestProperty("Content-Type", "application/json; utf-8");
        conn.setRequestProperty("Accept", "application/json");
        conn.setDoOutput(true);
        conn.setConnectTimeout(CONNECT_TIMEOUT_MS);
        conn.setReadTimeout(READ_TIMEOUT_MS);

        // Send the payload
        try (OutputStream os = conn.getOutputStream()) {
            byte[] input = jsonPayload.getBytes(StandardCharsets.UTF_8);
            os.write(input, 0, input.length);
        }

        // Read the response
        StringBuilder response = new StringBuilder();
        try (BufferedReader br = new BufferedReader(new InputStreamReader(conn.getInputStream(), StandardCharsets.UTF_8))) {
            String responseLine;
            while ((responseLine = br.readLine()) != null) {
                response.append(responseLine.trim());
            }
        }

        // Manually parse the action from the JSON response (e.g., {"action": 3})
        String jsonResponse = response.toString();
        int actionIndex = jsonResponse.indexOf("\"action\"");
        if (actionIndex == -1) {
            throw new Exception("Invalid response from server: " + jsonResponse);
        }
        
        // Find the number after "action":
        String sub = jsonResponse.substring(actionIndex + 8); // Skip past "action":
        sub = sub.replaceAll("[^\\d]", ""); // Remove non-numeric characters
        
        if (sub.isEmpty()) {
             throw new Exception("Could not parse action from server response: " + jsonResponse);
        }

        return Integer.parseInt(sub);
    }
    
    /**
     * Updates the 'currentImage' based on the action, so 'draw()' works.
     */
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
            // On STAY, keep the last image
        }
    }

    // --- Other required methods from Bot class ---

    @Override
    public void draw(Graphics g, int x, int y) {
        if (this.currentImage != null) {
            g.drawImage(this.currentImage, x, y, Bot.RADIUS * 2, Bot.RADIUS * 2, null);
        } else {
            // Fallback drawing if images fail to load
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
        return "PPO"; // Team name
    }

    @Override
    public String[] imageNames() {
        // Use the same roomba images as Drone for a placeholder
        return new String[]{"roomba_up.png", "roomba_down.png", "roomba_left.png", "roomba_right.png"};
    }

    @Override
    public void loadedImages(Image[] images) {
        if (images != null && images.length == 4) {
            this.botImages[0] = images[0]; // up
            this.botImages[1] = images[1]; // down
            this.botImages[2] = images[2]; // left
            this.botImages[3] = images[3]; // right
            this.currentImage = this.botImages[0]; // Default to 'up'
        }
    }

    @Override
    public String outgoingMessage() {
        return null; // This bot doesn't send messages
    }

    @Override
    public void incomingMessage(int botNum, String msg) {
        // This bot doesn't listen to messages
    }
}