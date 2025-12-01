# 관리자 웹 애플리케이션 아키텍처 정의서 및 시스템 설계서

**프로젝트명**: llm_chal_web (lc-app)  
**작성일**: 2025-11-24  
**버전**: 1.0  
**기반 시스템**: llm_chal_vlm (AI 서버)

---

## 목차

1. [시스템 개요](#1-시스템-개요)
2. [아키텍처 개요](#2-아키텍처-개요)
3. [기술 스택](#3-기술-스택)
4. [레이어드 아키텍처](#4-레이어드-아키텍처)
5. [도메인 모델](#5-도메인-모델)
6. [API 설계](#6-api-설계)
7. [외부 시스템 연동](#7-외부-시스템-연동)
8. [화면 구성](#8-화면-구성)
9. [보안 설계](#9-보안-설계)
10. [배포 구성](#10-배포-구성)

---

## 1. 시스템 개요

### 1.1 목적

llm_chal_web은 유사이미지 검색 솔루션(llm_chal_vlm)의 **관리자 전용 웹 애플리케이션**입니다. 
관리자가 제품, 불량 유형, 이미지, 매뉴얼을 관리하고 AI 모델 배포를 수행할 수 있는 인터페이스를 제공합니다.

### 1.2 핵심 기능

| 구분 | 기능 | 설명 |
|------|------|------|
| 제품 관리 | 제품 등록/조회 | 검사 대상 제품 정보 관리 |
| 불량 유형 관리 | 불량 유형 CRUD | 제품별 불량 유형 정의 |
| 매뉴얼 관리 | PDF 업로드/조회 | 대응 매뉴얼 문서 관리 |
| 이미지 관리 | 정상/불량 이미지 등록 | 학습 데이터셋 관리 |
| 모델 설정 | AI 모델 선택 | 검사/매칭/생성 모델 설정 |
| 서버 배포 | 임베딩/메모리뱅크 생성 | CLIP, PatchCore 배포 |
| 대시보드 | 통계 모니터링 | 시스템 현황 조회 |

### 1.3 시스템 관계도
```
┌─────────────────────────────────────────────────────────────────────┐
│                        전체 시스템 구성도                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   ┌─────────────────┐         ┌─────────────────┐                  │
│   │   관리자 브라우저   │         │   작업자 브라우저   │                  │
│   └────────┬────────┘         └────────┬────────┘                  │
│            │                           │                           │
│            ▼                           ▼                           │
│   ┌─────────────────┐         ┌─────────────────┐                  │
│   │  llm_chal_web   │◄───────►│  llm_chal_vlm   │                  │
│   │  (관리자 페이지)  │  HTTP    │  (AI 서버/작업자)  │                  │
│   │  Spring Boot    │         │  FastAPI        │                  │
│   │  Port: 8080     │         │  Port: 8000     │                  │
│   └────────┬────────┘         └────────┬────────┘                  │
│            │                           │                           │
│            ▼                           ▼                           │
│   ┌─────────────────────────────────────────────┐                  │
│   │              MariaDB Database               │                  │
│   │              (공용 데이터베이스)               │                  │
│   └─────────────────────────────────────────────┘                  │
│            │                                                       │
│            ▼                                                       │
│   ┌─────────────────────────────────────────────┐                  │
│   │         NCP Object Storage (dm-obs)         │                  │
│   │         (파일 저장소)                         │                  │
│   └─────────────────────────────────────────────┘                  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. 아키텍처 개요

### 2.1 아키텍처 패턴

**Layered Architecture (계층형 아키텍처)** 채택
```
┌─────────────────────────────────────────────────┐
│            Interfaces Layer (표현 계층)           │
│  ┌──────────────────┐  ┌──────────────────┐     │
│  │   Web Controller │  │   API Controller │     │
│  │   (Thymeleaf)    │  │   (REST)         │     │
│  └──────────────────┘  └──────────────────┘     │
├─────────────────────────────────────────────────┤
│           Application Layer (응용 계층)           │
│  ┌──────────────────────────────────────────┐   │
│  │  Service / Repository / Entity / Client  │   │
│  └──────────────────────────────────────────┘   │
├─────────────────────────────────────────────────┤
│          Infrastructure Layer (인프라 계층)        │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐  │
│  │ JPA Config │ │ S3 Config  │ │  Security  │  │
│  │            │ │ (NCP OBS)  │ │  Config    │  │
│  └────────────┘ └────────────┘ └────────────┘  │
└─────────────────────────────────────────────────┘
```

### 2.2 패키지 구조
```
src/main/java/kr/co/dimillion/lcapp/
├── LcAppApplication.java           # Spring Boot 메인
├── application/                    # 응용 계층
│   ├── Entity 클래스               # JPA Entity
│   │   ├── Product.java
│   │   ├── DefectType.java
│   │   ├── Manual.java
│   │   ├── Image.java
│   │   ├── ModelSettings.java
│   │   ├── SearchHistory.java
│   │   └── ResponseHistory.java
│   ├── Repository 인터페이스        # Spring Data JPA
│   │   ├── ProductRepository.java
│   │   ├── DefectTypeRepository.java
│   │   ├── ManualRepository.java
│   │   ├── ImageRepository.java
│   │   └── ...
│   ├── Service 클래스              # 비즈니스 로직
│   │   ├── ProductService.java
│   │   ├── ManualService.java
│   │   ├── ImageService.java
│   │   └── DefectTypeService.java
│   ├── AiServerClient.java         # AI 서버 HTTP 클라이언트
│   └── FileSystem.java             # Object Storage 파일 처리
│
├── interfaces/                     # 표현 계층
│   ├── web/                        # MVC 컨트롤러
│   │   ├── AdminController.java    # 관리자 기능 컨트롤러
│   │   ├── DashboardController.java
│   │   ├── LoginController.java
│   │   ├── FileApiController.java
│   │   ├── HealthController.java
│   │   └── HomeController.java
│   └── api/                        # REST API
│       └── DefectTypeRestController.java
│
└── infrastructure/                 # 인프라 계층
    ├── JpaConfig.java              # JPA 설정
    ├── HttpClientConfig.java       # WebClient 설정
    ├── NcpObjectStorageConfig.java # NCP S3 클라이언트
    └── security/                   # Spring Security
        └── SecurityConfig.java
```

---

## 3. 기술 스택

### 3.1 Backend

| 구분 | 기술 | 버전 |
|------|------|------|
| Language | Java | 21 |
| Framework | Spring Boot | 3.5.7 |
| Template Engine | Thymeleaf | - |
| ORM | Spring Data JPA | - |
| Security | Spring Security 6 | - |
| HTTP Client | Spring WebFlux (WebClient) | - |
| Build Tool | Gradle | 8.x |

### 3.2 Database

| 구분 | 기술 | 버전 |
|------|------|------|
| RDBMS | MariaDB | 10.5+ |
| Driver | mariadb-java-client | - |

### 3.3 Cloud Services

| 구분 | 서비스 | 용도 |
|------|--------|------|
| Object Storage | NCP Object Storage | 이미지/매뉴얼 파일 저장 |
| SDK | AWS SDK for Java v2 (S3 호환) | Object Storage 연동 |

### 3.4 Frontend

| 구분 | 기술 | 용도 |
|------|------|------|
| Template | Thymeleaf | SSR 렌더링 |
| CSS | TailwindCSS | 스타일링 |
| JavaScript | Vanilla JS | 클라이언트 로직 |

### 3.5 의존성 (build.gradle)
```groovy
dependencies {
    // Web
    implementation 'org.springframework.boot:spring-boot-starter-thymeleaf'
    implementation 'org.springframework.boot:spring-boot-starter-web'
    implementation 'org.springframework.boot:spring-boot-starter-webflux'
    
    // Security
    implementation 'org.springframework.boot:spring-boot-starter-security'
    implementation 'org.thymeleaf.extras:thymeleaf-extras-springsecurity6'
    
    // Database
    implementation 'org.springframework.boot:spring-boot-starter-data-jpa'
    runtimeOnly 'org.mariadb.jdbc:mariadb-java-client'
    
    // Cloud
    implementation 'software.amazon.awssdk:s3:2.20.50'
    
    // Utils
    compileOnly 'org.projectlombok:lombok'
    annotationProcessor 'org.projectlombok:lombok'
}
```

---

## 4. 레이어드 아키텍처

### 4.1 Interfaces Layer (표현 계층)

**역할**: HTTP 요청/응답 처리, 화면 렌더링

#### Web Controllers

| Controller | 경로 | 역할 |
|------------|------|------|
| AdminController | `/admin/**` | 관리자 기능 전체 |
| DashboardController | `/dashboard/**` | 대시보드 통계 |
| LoginController | `/login` | 로그인 처리 |
| HealthController | `/health` | ALB 헬스체크 |
| HomeController | `/` | 메인 페이지 |
| FileApiController | `/api/files/**` | 파일 다운로드 |

#### AdminController 주요 메서드

| 메서드 | HTTP | 경로 | 기능 |
|--------|------|------|------|
| productManagement | GET | /admin/product-management | 제품 목록 조회 |
| product | POST | /admin/product-management/product | 제품 등록 |
| manualManagement | GET | /admin/manual-management | 매뉴얼 목록 |
| manual | POST/DELETE | /admin/manual-management/manual | 매뉴얼 등록/삭제 |
| defectManagement | GET | /admin/defect-type-management | 불량유형 목록 |
| createDefectType | POST/PUT/DELETE | /admin/defect-type-management/defect-type | 불량유형 CRUD |
| normalImageManagement | GET | /admin/normal-image-management | 정상이미지 목록 |
| createNormalImage | POST/DELETE | /admin/normal-image-management/image | 정상이미지 등록/삭제 |
| defectImageManagement | GET | /admin/defect-image-management | 불량이미지 목록 |
| createDefectImage | POST/DELETE | /admin/defect-image-management/image | 불량이미지 등록/삭제 |
| modelSettings | GET/POST | /admin/model-settings | 모델 설정 |
| embedding | GET/POST | /admin/embedding/** | CLIP 임베딩 배포 |
| patchcore | GET/POST | /admin/patchcore | PatchCore 배포 |

### 4.2 Application Layer (응용 계층)

**역할**: 비즈니스 로직, 도메인 모델, 데이터 접근

#### Entities (도메인 모델)

| Entity | 테이블 | 설명 |
|--------|--------|------|
| Product | products | 제품 정보 |
| DefectType | defect_types | 불량 유형 |
| Manual | manuals | 매뉴얼 파일 |
| Image | images | 이미지 파일 |
| ModelSettings | model_settings | AI 모델 설정 |
| SearchHistory | search_history | 검색 이력 |
| ResponseHistory | response_history | 응답 이력 |

#### Services

| Service | 역할 |
|---------|------|
| ProductService | 제품 생성 로직 |
| ManualService | 매뉴얼 저장/비활성화 |
| ImageService | 이미지 삭제 |
| DefectTypeService | 불량유형 수정/삭제 |

#### AiServerClient

AI 서버(llm_chal_vlm)와의 HTTP 통신 인터페이스:
```java
@HttpExchange("/api")
public interface AiServerClient {
    @PostExchange("/admin/manual/sync-manual")
    Mono<Void> syncManuals();
    
    @PostExchange("/admin/defect-type/refresh-mapping")
    Mono<Void> refreshDefectTypeMapping();
    
    @PostExchange("/admin/image/sync-normal")
    Mono<Void> syncNormalImages();
    
    @PostExchange("/admin/image/sync-defect")
    Mono<Void> syncDefectImages();
    
    @PostExchange("/admin/deployment/v2/clip/normal")
    Mono<Void> deployClipNormal();
    
    @PostExchange("/admin/deployment/v2/clip/defect")
    Mono<Void> deployClipDefect();
    
    @PostExchange("/admin/deployment/patchcore")
    Mono<Void> deployPatchCore();
}
```

#### FileSystem

NCP Object Storage 파일 업로드/다운로드:
```java
@Service
public class FileSystem {
    public String uploadFile(MultipartFile file, String subFolder);
    public byte[] downloadFile(String subFolder, String fileName);
}
```

### 4.3 Infrastructure Layer (인프라 계층)

**역할**: 외부 시스템 연동, 설정

| Config | 역할 |
|--------|------|
| JpaConfig | JPA Auditing 설정 |
| HttpClientConfig | AI 서버용 WebClient 설정 |
| NcpObjectStorageConfig | NCP S3 클라이언트 설정 |
| SecurityConfig | Spring Security 설정 |

---

## 5. 도메인 모델

### 5.1 Entity 관계도 (ERD)
```
┌─────────────┐       ┌─────────────┐
│   Product   │       │ DefectType  │
│─────────────│       │─────────────│
│ id (PK)     │◄──┐   │ id (PK)     │
│ code        │   │   │ product_id  │──►┐
│ name        │   │   │ code        │   │
│ description │   │   │ name_ko     │   │
│ created_at  │   │   │ name_en     │   │
│ updated_at  │   └───│ description │   │
└─────────────┘       │ used        │   │
       │              └─────────────┘   │
       │                     │          │
       ▼                     ▼          │
┌─────────────┐       ┌─────────────┐   │
│   Manual    │       │    Image    │   │
│─────────────│       │─────────────│   │
│ id (PK)     │       │ id (PK)     │   │
│ product_id  │──►    │ product_id  │──►┘
│ name        │       │ defect_type_id │──►
│ path        │       │ type        │
│ size        │       │ filename    │
│ indexed     │       │ filepath    │
│ used        │       │ filesize    │
│ created_at  │       │ used        │
└─────────────┘       │ created_at  │
                      └─────────────┘

┌─────────────────┐
│  ModelSettings  │
│─────────────────│
│ id (PK)         │
│ defect_inspection │
│ similarity_match  │
│ gen_ai          │
│ created_at      │
└─────────────────┘
```

### 5.2 주요 Entity 상세

#### Product
```java
@Entity
@Table(name = "products")
public class Product extends BaseEntity {
    @Id @GeneratedValue
    private Integer id;
    private String code;        // 제품 코드
    private String name;        // 제품명
    private String description; // 설명
}
```

#### DefectType
```java
@Entity
@Table(name = "defect_types")
public class DefectType extends BaseEntity {
    @Id @GeneratedValue
    private Integer id;
    
    @ManyToOne
    private Product product;
    
    private String code;        // 불량 코드 (영문)
    private String nameKo;      // 불량명 (한글)
    private String nameEn;      // 불량명 (영문)
    private String description;
    private boolean used;       // 사용 여부
}
```

#### Image
```java
@Entity
@Table(name = "images")
public class Image extends BaseEntity {
    @Id @GeneratedValue
    private Integer id;
    
    @ManyToOne
    private Product product;
    
    @ManyToOne
    private DefectType defectType;
    
    private String type;        // "normal" | "defect"
    private String filename;
    private String filepath;    // Object Storage 경로
    private Long filesize;
    private boolean used;
}
```

#### ModelSettings
```java
@Entity
@Table(name = "model_settings")
public class ModelSettings {
    @Id @GeneratedValue
    private Integer id;
    
    @Enumerated(EnumType.STRING)
    private DefectInspection defectInspection;  // PATCHCORE
    
    @Enumerated(EnumType.STRING)
    private SimilarityMatch similarityMatch;    // CLIP_VIT_B_32
    
    @Enumerated(EnumType.STRING)
    private GenAi genAi;                        // CLAUDE_3_5_SONNET
}
```

---

## 6. API 설계

### 6.1 관리자 페이지 API (Server-Side Rendering)

| Method | URL | 기능 | 응답 |
|--------|-----|------|------|
| GET | /admin | 관리자 메인 | admin.html |
| GET | /admin/product-management | 제품 목록 | product-management.html |
| POST | /admin/product-management/product | 제품 등록 | Redirect |
| GET | /admin/manual-management | 매뉴얼 목록 | manual-management.html |
| POST | /admin/manual-management/manual | 매뉴얼 업로드 | Redirect |
| DELETE | /admin/manual-management/manual | 매뉴얼 삭제 | Redirect |
| GET | /admin/defect-type-management | 불량유형 목록 | defect-type-management.html |
| POST | /admin/defect-type-management/defect-type | 불량유형 등록 | Redirect |
| PUT | /admin/defect-type-management/defect-type | 불량유형 수정 | Redirect |
| DELETE | /admin/defect-type-management/defect-type | 불량유형 삭제 | Redirect |
| GET | /admin/normal-image-management | 정상이미지 목록 | normal-image-management.html |
| POST | /admin/normal-image-management/image | 정상이미지 업로드 | Redirect |
| DELETE | /admin/normal-image-management/image | 정상이미지 삭제 | Redirect |
| GET | /admin/defect-image-management | 불량이미지 목록 | defect-image-management.html |
| POST | /admin/defect-image-management/image | 불량이미지 업로드 | Redirect |
| DELETE | /admin/defect-image-management/image | 불량이미지 삭제 | Redirect |
| GET | /admin/model-settings | 모델 설정 조회 | model-settings.html |
| POST | /admin/model-settings | 모델 설정 저장 | Redirect |
| GET | /admin/embedding | 임베딩 페이지 | embedding.html |
| POST | /admin/embedding/normal | 정상이미지 임베딩 실행 | Redirect |
| POST | /admin/embedding/defect | 불량이미지 임베딩 실행 | Redirect |
| GET | /admin/patchcore | PatchCore 페이지 | patchcore.html |
| POST | /admin/patchcore | PatchCore 배포 실행 | Redirect |

### 6.2 REST API

| Method | URL | 기능 | 응답 |
|--------|-----|------|------|
| GET | /api/defect-types | 제품별 불량유형 조회 | JSON |
| GET | /api/files/download/{folder}/{filename} | 파일 다운로드 | Binary |
| GET | /health | 헬스체크 | "OK" |

### 6.3 대시보드 API

| Method | URL | 기능 |
|--------|-----|------|
| GET | /dashboard | 대시보드 메인 |
| GET | /dashboard/api/inspection-trend | 검사 트렌드 데이터 |
| GET | /dashboard/api/defect-distribution | 불량 분포 데이터 |
| GET | /dashboard/api/product-trend | 제품별 트렌드 |

---

## 7. 외부 시스템 연동

### 7.1 AI 서버 (llm_chal_vlm) 연동

**통신 방식**: HTTP (WebClient, Reactive)
```
┌─────────────────┐                 ┌─────────────────┐
│  llm_chal_web   │     HTTP/POST   │  llm_chal_vlm   │
│  (Spring Boot)  │ ───────────────►│  (FastAPI)      │
│                 │                 │                 │
│  AiServerClient │                 │  Admin API      │
└─────────────────┘                 └─────────────────┘
```

**연동 API 목록**:

| 관리자 페이지 액션 | AI 서버 API | 설명 |
|-------------------|-------------|------|
| 매뉴얼 업로드/삭제 | POST /api/admin/manual/sync-manual | 매뉴얼 동기화 |
| 불량유형 변경 | POST /api/admin/defect-type/refresh-mapping | 매핑 갱신 |
| 정상이미지 변경 | POST /api/admin/image/sync-normal | 정상이미지 동기화 |
| 불량이미지 변경 | POST /api/admin/image/sync-defect | 불량이미지 동기화 |
| 정상이미지 임베딩 | POST /api/admin/deployment/v2/clip/normal | CLIP 인덱스 생성 |
| 불량이미지 임베딩 | POST /api/admin/deployment/v2/clip/defect | CLIP 인덱스 생성 |
| PatchCore 배포 | POST /api/admin/deployment/patchcore | 메모리뱅크 생성 |

**설정 (application.yml)**:
```yaml
ai-server-host: ${AI_SERVER_HOST}  # 예: http://localhost:8000
```

### 7.2 NCP Object Storage 연동

**통신 방식**: S3 호환 API (AWS SDK)

**버킷 구조**:
```
dm-obs/
├── menual_store/          # 매뉴얼 PDF 저장
│   └── {uuid}.pdf
├── ok_image/              # 정상 이미지 저장
│   └── {uuid}.jpg
└── def_split/             # 불량 이미지 저장
    └── {uuid}.jpg
```

**설정 (application.yml)**:
```yaml
cloud:
  ncp:
    credentials:
      access-key: ${NCP_ACCESS_KEY}
      secret-key: ${NCP_SECRET_KEY}
    object-storage:
      bucket: ${NCP_BUCKET}  # dm-obs
```

**NcpObjectStorageConfig.java**:
```java
@Bean
public S3Client s3Client() {
    return S3Client.builder()
        .endpointOverride(URI.create("https://kr.object.ncloudstorage.com"))
        .region(Region.of("kr-standard"))
        .credentialsProvider(StaticCredentialsProvider.create(
            AwsBasicCredentials.create(accessKey, secretKey)))
        .build();
}
```

### 7.3 MariaDB 연동

**공용 데이터베이스**: llm_chal_vlm과 동일한 DB 사용

**설정 (application.yml)**:
```yaml
spring:
  datasource:
    driver-class-name: org.mariadb.jdbc.Driver
    url: jdbc:mariadb://${DB_HOST}:${DB_PORT}/${DB_NAME}
    username: ${DB_USER}
    password: ${DB_PASSWORD}
  jpa:
    hibernate:
      ddl-auto: validate  # 스키마 검증만 (수정 안함)
```

---

## 8. 화면 구성

### 8.1 화면 목록

| 화면명 | 파일명 | 경로 | 설명 |
|--------|--------|------|------|
| 로그인 | login.html | /login | 인증 페이지 |
| 메인 | index.html | / | 시작 페이지 |
| 관리자 메인 | admin.html | /admin | 관리자 메뉴 |
| 대시보드 | dashboard.html | /dashboard | 통계 현황 |
| 제품 관리 | product-management.html | /admin/product-management | 제품 CRUD |
| 매뉴얼 관리 | manual-management.html | /admin/manual-management | 매뉴얼 업로드 |
| 불량유형 관리 | defect-type-management.html | /admin/defect-type-management | 불량유형 CRUD |
| 정상이미지 관리 | normal-image-management.html | /admin/normal-image-management | 정상이미지 업로드 |
| 불량이미지 관리 | defect-image-management.html | /admin/defect-image-management | 불량이미지 업로드 |
| 모델 설정 | model-settings.html | /admin/model-settings | AI 모델 선택 |
| 임베딩 배포 | embedding.html | /admin/embedding | CLIP 배포 |
| PatchCore 배포 | patchcore.html | /admin/patchcore | PatchCore 배포 |

### 8.2 화면 구조
```
┌─────────────────────────────────────────────────────────────────┐
│                        Header (네비게이션)                        │
├─────────────────────────────────────────────────────────────────┤
│ ┌─────────────────┬───────────────────────────────────────────┐ │
│ │                 │                                           │ │
│ │   사이드바       │              메인 콘텐츠                    │ │
│ │   (메뉴)        │                                           │ │
│ │                 │                                           │ │
│ │  - 대시보드      │                                           │ │
│ │  - 제품관리      │                                           │ │
│ │  - 매뉴얼관리    │                                           │ │
│ │  - 불량유형관리  │                                           │ │
│ │  - 정상이미지    │                                           │ │
│ │  - 불량이미지    │                                           │ │
│ │  - 모델설정      │                                           │ │
│ │  - 서버배포      │                                           │ │
│ │                 │                                           │ │
│ └─────────────────┴───────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### 8.3 Thymeleaf Fragment 구조
```
templates/
├── _fragments/
│   ├── head.html      # <head> 공통
│   ├── header.html    # 상단 네비게이션
│   ├── sidebar.html   # 좌측 메뉴
│   └── scripts.html   # 공통 JavaScript
├── admin.html
├── dashboard.html
└── ...
```

---

## 9. 보안 설계

### 9.1 인증/인가

**Spring Security 기반 인증**:
- Form Login 방식
- 세션 기반 인증
- CSRF 보호 활성화

**권한 모델**:
```java
public enum Role {
    ADMIN,   // 관리자
    USER     // 일반 사용자
}
```

### 9.2 보안 설정
```java
@Configuration
@EnableWebSecurity
public class SecurityConfig {
    @Bean
    public SecurityFilterChain filterChain(HttpSecurity http) {
        return http
            .authorizeHttpRequests(auth -> auth
                .requestMatchers("/health").permitAll()
                .requestMatchers("/css/**", "/js/**", "/images/**").permitAll()
                .requestMatchers("/login").permitAll()
                .requestMatchers("/admin/**").hasRole("ADMIN")
                .anyRequest().authenticated()
            )
            .formLogin(form -> form
                .loginPage("/login")
                .defaultSuccessUrl("/admin")
            )
            .build();
    }
}
```

### 9.3 파일 업로드 보안

- 파일 크기 제한: 50MB (개별), 100MB (요청 전체)
- 파일명 UUID 변환으로 경로 탐색 공격 방지
- Object Storage에 Public Read ACL 적용

---

## 10. 배포 구성

### 10.1 인프라 구성
```
┌─────────────────────────────────────────────────────────────────┐
│                    Naver Cloud Platform                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                     VPC (dm-vpc)                         │   │
│  │                   10.200.0.0/16                          │   │
│  │  ┌─────────────────────────────────────────────────────┐ │   │
│  │  │            Load Balancer Subnet                     │ │   │
│  │  │            (dm-lb-sub: 10.200.1.0/24)              │ │   │
│  │  │  ┌────────────┐                                     │ │   │
│  │  │  │    ALB     │  Port 80 → 8080                    │ │   │
│  │  │  └─────┬──────┘                                     │ │   │
│  │  └────────┼────────────────────────────────────────────┘ │   │
│  │           │                                              │   │
│  │  ┌────────┼────────────────────────────────────────────┐ │   │
│  │  │        ▼        Private Subnet                      │ │   │
│  │  │               (dm-pri-sub: 10.200.3.0/24)           │ │   │
│  │  │  ┌────────────────────────────────────┐             │ │   │
│  │  │  │          GPU Server                │             │ │   │
│  │  │  │  ┌──────────────┐ ┌──────────────┐ │             │ │   │
│  │  │  │  │llm_chal_web │ │llm_chal_vlm │ │             │ │   │
│  │  │  │  │ Port: 8080  │ │ Port: 8000  │ │             │ │   │
│  │  │  │  └──────────────┘ └──────────────┘ │             │ │   │
│  │  │  └────────────────────────────────────┘             │ │   │
│  │  └─────────────────────────────────────────────────────┘ │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────┐  ┌─────────────────┐                      │
│  │  Object Storage │  │    MariaDB      │                      │
│  │    (dm-obs)     │  │                 │                      │
│  └─────────────────┘  └─────────────────┘                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 10.2 배포 스크립트 (deploy.sh)
```bash
#!/bin/bash

echo "=== 빌드 시작 ==="
./gradlew clean build -x test

echo "=== 기존 프로세스 종료 ==="
pkill -f 'java.*lc-app' || true

echo "=== 애플리케이션 시작 ==="
nohup java -jar build/libs/lc-app-0.0.1-SNAPSHOT.jar \
  --server.port=8080 \
  > app.log 2>&1 &

echo "=== 배포 완료 ==="
```

### 10.3 환경 변수
```bash
# Database
export DB_HOST=localhost
export DB_PORT=3306
export DB_NAME=llm_chal_db
export DB_USER=dmillion
export DB_PASSWORD=****

# NCP Object Storage
export NCP_ACCESS_KEY=****
export NCP_SECRET_KEY=****
export NCP_BUCKET=dm-obs

# AI Server
export AI_SERVER_HOST=http://localhost:8000
```

### 10.4 접속 정보

| 서비스 | URL | 용도 |
|--------|-----|------|
| 관리자 페이지 | http://dm-alb-112319279-991b4e0889c4.kr.lb.naverncp.com:8080 | 외부 접속 (ALB) |
| 내부 접속 | http://10.200.3.x:8080 | 서버 내부 |
| 헬스체크 | /health | ALB 상태 확인 |

---

## 부록: 주요 처리 흐름

### A. 이미지 업로드 흐름
```
1. 사용자 → 관리자 페이지: 이미지 파일 업로드
2. AdminController: MultipartFile 수신
3. FileSystem: NCP Object Storage에 업로드
   → 반환: "/ok_image/{uuid}.jpg"
4. ImageRepository: DB에 메타데이터 저장
5. AiServerClient: AI 서버에 동기화 요청 (비동기)
6. AI 서버: Object Storage에서 이미지 다운로드 → 로컬 배치
7. 사용자에게 성공 응답 (Redirect)
```

### B. 서버 배포 흐름 (CLIP 임베딩)
```
1. 사용자 → 관리자 페이지: 임베딩 실행 버튼 클릭
2. AdminController: AiServerClient.deployClipNormal() 호출
3. AiServerClient → AI 서버: POST /api/admin/deployment/v2/clip/normal
4. AI 서버: 
   - Object Storage에서 이미지 다운로드
   - CLIP 임베딩 생성
   - FAISS 인덱스 저장
5. 사용자에게 성공 응답 (Redirect with ?normalSuccess)
```

---

**검토자**: dhkim  
**최종 수정일**: 2025-11-24