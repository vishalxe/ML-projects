/**
 * Gmail Add-on for Email Auto-Categorization
 * This script runs when Gmail is opened and categorizes emails automatically
 */

// Configuration
const CATEGORIES = {
  promotions: "📂 Promotions",
  notifications: "🔔 Notifications",
  important: "📌 Important",
  jobs: "💼 Jobs",
  spam: "🚫 Spam",
  alerts: "⚠️ Alerts",
};

const ALERT_KEYWORDS = [
  "expire",
  "expiring",
  "ending",
  "last date",
  "final day",
  "today only",
  "due in",
  "offer ends",
  "deadline",
  "last chance",
];

/**
 * Main function called when Gmail add-on is opened
 */
function onGmailCompose(e) {
  return createAddonCard();
}

function onGmailOpen(e) {
  return createAddonCard();
}

/**
 * Creates the main add-on card
 */
function createAddonCard() {
  const card = CardService.newCardBuilder()
    .setHeader(
      CardService.newCardHeader()
        .setTitle("📧 Email Auto-Categorizer")
        .setSubtitle("Automatically categorize your emails")
    )
    .addSection(createMainSection())
    .build();

  return [card];
}

/**
 * Creates the main section with controls
 */
function createMainSection() {
  const section = CardService.newCardSection()
    .setHeader("Email Categorization")
    .addWidget(
      CardService.newTextParagraph().setText(
        "This add-on will automatically categorize your emails based on content analysis."
      )
    )
    .addWidget(
      CardService.newTextInput()
        .setFieldName("emailText")
        .setTitle("Email Content")
        .setMultiline(true)
        .setValue("Paste email content here...")
    )
    .addWidget(
      CardService.newTextButton()
        .setText("🚀 Categorize Email")
        .setOnClickAction(
          CardService.newAction().setFunctionName("categorizeEmail")
        )
    )
    .addWidget(
      CardService.newTextButton()
        .setText("📊 View Categories")
        .setOnClickAction(
          CardService.newAction().setFunctionName("showCategories")
        )
    )
    .addWidget(
      CardService.newTextButton()
        .setText("⚙️ Auto-Categorize Inbox")
        .setOnClickAction(
          CardService.newAction().setFunctionName("autoCategorizeInbox")
        )
    );

  return section;
}

/**
 * Categorizes a single email
 */
function categorizeEmail(e) {
  const emailText = e.formInputs.emailText[0];

  if (!emailText || emailText.trim() === "") {
    return createErrorCard("Please enter email content to categorize.");
  }

  const category = predictCategory(emailText);
  const confidence = calculateConfidence(emailText, category);

  return createResultCard(category, confidence, emailText);
}

/**
 * Shows categorized emails by category
 */
function showCategories(e) {
  const threads = GmailApp.getInboxThreads(0, 50); // Get last 50 emails
  const categorizedEmails = {};

  // Initialize categories
  Object.keys(CATEGORIES).forEach((cat) => {
    categorizedEmails[cat] = [];
  });

  // Categorize emails
  threads.forEach((thread) => {
    const messages = thread.getMessages();
    messages.forEach((message) => {
      const subject = message.getSubject();
      const body = message.getPlainBody();
      const emailContent = `${subject} ${body}`.substring(0, 500); // Limit content

      const category = predictCategory(emailContent);
      categorizedEmails[category].push({
        subject: subject,
        snippet: body.substring(0, 100) + "...",
        date: message.getDate(),
      });
    });
  });

  return createCategoriesCard(categorizedEmails);
}

/**
 * Auto-categorizes emails in inbox and applies labels
 */
function autoCategorizeInbox(e) {
  const threads = GmailApp.getInboxThreads(0, 100); // Get last 100 emails
  let processedCount = 0;

  threads.forEach((thread) => {
    const messages = thread.getMessages();
    messages.forEach((message) => {
      const subject = message.getSubject();
      const body = message.getPlainBody();
      const emailContent = `${subject} ${body}`.substring(0, 500);

      const category = predictCategory(emailContent);

      // Create or get label
      let label = GmailApp.getUserLabelByName(CATEGORIES[category]);
      if (!label) {
        label = GmailApp.createLabel(CATEGORIES[category]);
      }

      // Apply label to thread
      thread.addLabel(label);
      processedCount++;
    });
  });

  return createSuccessCard(
    `Successfully categorized ${processedCount} emails!`
  );
}

/**
 * Simple category prediction based on keywords
 * In a real implementation, you'd load your trained model here
 */
function predictCategory(emailText) {
  const text = emailText.toLowerCase();

  // Check for alerts first
  for (const keyword of ALERT_KEYWORDS) {
    if (text.includes(keyword)) {
      return "alerts";
    }
  }

  // Simple keyword-based categorization
  if (
    text.includes("job") ||
    text.includes("career") ||
    text.includes("resume") ||
    text.includes("application")
  ) {
    return "jobs";
  }

  if (
    text.includes("spam") ||
    text.includes("scam") ||
    text.includes("congratulations") ||
    text.includes("free gift")
  ) {
    return "spam";
  }

  if (
    text.includes("promotion") ||
    text.includes("discount") ||
    text.includes("offer") ||
    text.includes("deal")
  ) {
    return "promotions";
  }

  if (
    text.includes("notification") ||
    text.includes("alert") ||
    text.includes("security") ||
    text.includes("password")
  ) {
    return "notifications";
  }

  if (
    text.includes("important") ||
    text.includes("urgent") ||
    text.includes("appointment") ||
    text.includes("bill")
  ) {
    return "important";
  }

  // Default to notifications
  return "notifications";
}

/**
 * Calculate confidence score (simplified)
 */
function calculateConfidence(emailText, category) {
  // This is a simplified confidence calculation
  // In a real implementation, you'd use your trained model's probability scores
  return Math.random() * 0.4 + 0.6; // Random between 0.6-1.0
}

/**
 * Creates result card showing categorization
 */
function createResultCard(category, confidence, emailText) {
  const card = CardService.newCardBuilder()
    .setHeader(
      CardService.newCardHeader().setTitle("📧 Email Categorization Result")
    )
    .addSection(
      CardService.newCardSection()
        .addWidget(
          CardService.newKeyValue()
            .setTopLabel("Predicted Category")
            .setContent(CATEGORIES[category])
        )
        .addWidget(
          CardService.newKeyValue()
            .setTopLabel("Confidence")
            .setContent(`${Math.round(confidence * 100)}%`)
        )
        .addWidget(
          CardService.newTextParagraph().setText(
            `Email: ${emailText.substring(0, 200)}...`
          )
        )
    )
    .build();

  return [card];
}

/**
 * Creates categories card showing all categorized emails
 */
function createCategoriesCard(categorizedEmails) {
  const card = CardService.newCardBuilder()
    .setHeader(CardService.newCardHeader().setTitle("📊 Email Categories"))
    .build();

  Object.keys(categorizedEmails).forEach((category) => {
    const emails = categorizedEmails[category];
    if (emails.length > 0) {
      const section = CardService.newCardSection().setHeader(
        CATEGORIES[category] + ` (${emails.length})`
      );

      emails.slice(0, 5).forEach((email) => {
        // Show max 5 per category
        section.addWidget(
          CardService.newKeyValue()
            .setTopLabel(email.subject)
            .setContent(email.snippet)
        );
      });

      card.addSection(section);
    }
  });

  return [card];
}

/**
 * Creates success card
 */
function createSuccessCard(message) {
  const card = CardService.newCardBuilder()
    .setHeader(CardService.newCardHeader().setTitle("✅ Success"))
    .addSection(
      CardService.newCardSection().addWidget(
        CardService.newTextParagraph().setText(message)
      )
    )
    .build();

  return [card];
}

/**
 * Creates error card
 */
function createErrorCard(message) {
  const card = CardService.newCardBuilder()
    .setHeader(CardService.newCardHeader().setTitle("❌ Error"))
    .addSection(
      CardService.newCardSection().addWidget(
        CardService.newTextParagraph().setText(message)
      )
    )
    .build();

  return [card];
}

