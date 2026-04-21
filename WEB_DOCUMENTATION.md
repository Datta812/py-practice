# Cattle Breed Classification — Web Interface Documentation

## Overview

The Bharat Pashudhan App web interface is a modern, responsive single-page application designed for Field Level Workers (FLWs) to perform AI-powered cattle and buffalo breed identification and animal registration. The interface integrates with a Flask REST API backend that uses ONNX Runtime and a ResNet-50 model for breed classification.

## Architecture

- **Frontend**: Vanilla HTML5, CSS3, JavaScript (ES6+)
- **Backend**: Flask REST API with ONNX Runtime
- **AI Model**: ResNet-50 ONNX model (vishnuamar/cattle-breed-classifier)
- **Styling**: Custom CSS with CSS Variables for theming
- **Responsiveness**: Mobile-first design with breakpoints

## Core Features

### 1. AI Breed Scanner

**Location**: Main tool section (ID: `tool-section`)

**Functionality**:
- Image upload via drag-and-drop or file browser
- Sample breed selection for testing
- Real-time AI analysis with progress indicators
- Confidence scoring and alternative breed suggestions
- One-click application to registration form

**Supported Image Formats**:
- JPEG, PNG, WebP
- Maximum file size: 10 MB
- Recommended: Clear side/front view for best accuracy

**API Integration**:
- POST `/api/predict` for real image analysis
- GET `/api/predict/demo` for sample predictions
- Automatic fallback to demo mode when API is offline

### 2. Animal Registration Form

**Location**: Registration section (ID: `register-section`)

**Form Tabs**:
1. **Basic Info**: Animal tag ID, species, sex, age, DOB, location
2. **Breed & Physical**: AI-detected breed, confirmed breed, body weight, color, purpose
3. **Health & Nutrition**: Vaccination, BCS, milk yield, feeding system, remarks
4. **Owner Details**: Owner info, contact, location, FLW ID

**Features**:
- AI breed auto-fill from scanner results
- Form validation and data persistence
- Submission tracking and history
- Clear/reset functionality

### 3. Breed Reference Database

**Location**: Breeds section (ID: `breeds-section`)

**Features**:
- Interactive breed cards with filtering
- Detailed breed information modals
- Filter by: All, Cattle, Buffalo
- Click-to-view full breed specifications

**Supported Breeds**: 10 Indian breeds (5 cattle, 5 buffalo)

### 4. Registration History

**Location**: History section (ID: `history-section`)

**Features**:
- Session-based registration tracking
- Table view with sortable columns
- Method tracking (AI Verified vs Manual)
- Export-ready data structure

## User Interface Components

### Header
- **Logo**: App branding with government affiliation
- **API Status Indicator**: Real-time backend connectivity status
- **Navigation Pills**: Government initiative badges

### Hero Section
- **Statistics**: Key metrics (10 breeds, 94%+ accuracy, <50ms inference)
- **Call-to-Actions**: Direct links to scanner and database
- **Phone Mockup**: Visual representation of mobile usage

### Navigation Flow
1. **Landing** → Hero section with overview
2. **AI Scanner** → Upload/analyze → Apply to form
3. **Registration Form** → Fill details → Submit
4. **Breed Database** → Browse/reference → Modal details
5. **History** → View submitted registrations

## Technical Implementation

### CSS Architecture
- **CSS Variables**: Centralized color scheme and spacing
- **Component Classes**: Reusable UI components
- **Responsive Design**: Mobile-first with desktop enhancements
- **Animations**: Smooth transitions and loading states

### JavaScript Functionality

#### Core Functions
- `checkApi()`: Backend health monitoring
- `runScan()`: Main AI analysis workflow
- `showResult()`: Result display with confidence visualization
- `applyBreed()`: Form auto-population from AI results
- `submitForm()`: Registration submission and history tracking

#### State Management
- `apiOnline`: Backend connectivity status
- `scanReady`: Upload/analysis readiness
- `lastResult`: Current AI prediction data
- `regCount`: Session registration counter

#### Event Handlers
- File upload: Drag-and-drop and file input
- Sample selection: Breed preset buttons
- Form navigation: Tab switching
- Modal interactions: Breed detail popups

### API Communication

#### Endpoints Used
- `GET /api/health`: Backend status check
- `POST /api/predict`: Image analysis
- `GET /api/predict/demo`: Demo predictions
- `GET /api/breeds`: Breed metadata (future)

#### Error Handling
- Automatic fallback to demo mode
- User-friendly error messages
- Graceful degradation for offline usage

## User Experience Design

### Design Principles
- **Accessibility**: High contrast, readable fonts, keyboard navigation
- **Mobile-First**: Optimized for field worker devices
- **Progressive Enhancement**: Works without JavaScript
- **Visual Hierarchy**: Clear information architecture

### Color Scheme
- **Primary Green**: #0d4a28 (dark) to #2d9e5f (bright)
- **Accent Colors**: Amber (#e8930a), Red (#c0392b), Blue (#1a5fa8)
- **Neutral Grays**: Text hierarchy and borders
- **Semantic Colors**: Success (green), warning (amber), error (red)

### Typography
- **Primary Font**: Sora (Google Fonts)
- **Hierarchy**: 42px (hero) → 22px (section) → 15px (card) → 13px (body)
- **Weights**: 300 (light), 400 (regular), 500 (medium), 600 (semibold), 700 (bold)

## Performance Considerations

### Optimization Features
- **Lazy Loading**: Images and heavy components
- **Efficient DOM**: Minimal reflows and repaints
- **API Caching**: Model and metadata caching
- **Progressive Loading**: Content appears as ready

### Browser Support
- **Modern Browsers**: Chrome 90+, Firefox 88+, Safari 14+, Edge 90+
- **Mobile Browsers**: iOS Safari, Chrome Mobile
- **Fallbacks**: Graceful degradation for older browsers

## Accessibility Features

### WCAG Compliance
- **Color Contrast**: WCAG AA compliant ratios
- **Keyboard Navigation**: Full keyboard accessibility
- **Screen Readers**: Semantic HTML and ARIA labels
- **Focus Management**: Visible focus indicators

### Inclusive Design
- **Font Scaling**: Responsive typography
- **Touch Targets**: Minimum 44px touch targets
- **Error Prevention**: Form validation and confirmation
- **Clear Language**: Simple, field-appropriate terminology

## Development and Maintenance

### File Structure
```
static/
├── index.html          # Main application
└── (future assets)
```

### Code Organization
- **Inline CSS**: Component-scoped styling
- **Inline JS**: Application logic and event handlers
- **Modular Functions**: Single-responsibility functions
- **Constants**: Centralized configuration

### Testing Recommendations
- **Cross-browser testing** on target devices
- **API integration testing** for backend changes
- **User acceptance testing** with FLWs
- **Performance testing** on low-end devices

## Future Enhancements

### Planned Features
- **Offline Mode**: Service worker caching
- **Bulk Upload**: Multiple image processing
- **Advanced Filters**: Location-based breed suggestions
- **Export Functionality**: CSV/PDF generation
- **Multi-language Support**: Regional language options

### Technical Improvements
- **Component Framework**: Vue.js/React migration
- **Build System**: Webpack/Vite bundling
- **Testing Framework**: Jest/Cypress integration
- **Performance Monitoring**: Real user monitoring

## Deployment and Hosting

### Requirements
- **Web Server**: Static file hosting (Nginx, Apache, CDN)
- **HTTPS**: SSL certificate for security
- **Backend**: Flask API on same domain or CORS-enabled
- **Caching**: CDN for static assets

### Configuration
- **API_BASE**: Backend URL configuration
- **Feature Flags**: Demo mode toggles
- **Environment Variables**: Production settings

## Support and Training

### User Training
- **Quick Start Guide**: 5-minute onboarding
- **Video Tutorials**: Step-by-step workflows
- **Field Manuals**: Offline documentation
- **Help System**: In-app contextual help

### Technical Support
- **Error Logging**: Client-side error reporting
- **Performance Metrics**: Usage analytics
- **Feedback System**: User improvement suggestions
- **Version Updates**: Seamless deployment strategy

---

**Version**: 1.0.0
**Last Updated**: April 21, 2026
**Developed for**: Ministry of Fisheries, Animal Husbandry & Dairying, Government of India</content>
<parameter name="filePath">c:\Users\kusad\OneDrive\Desktop\py practice\WEB_DOCUMENTATION.md