// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __PUJ__Linear__h__
#define __PUJ__Linear__h__

#include <PUJ/Traits.h>
#include <PUJ/Model/BaseCost.h>

namespace PUJ
{
  namespace Model
  {
    /**
     */
    template< class _TScalar, class _TTraits = PUJ::Traits< _TScalar > >
    class PUJ_ML_EXPORT Linear
    {
    public:
      PUJ_TraitsMacro( Linear );

    protected:
      using _TBaseCost = PUJ::Model::BaseCost< Self >;
      using _TCol = Eigen::Map< TCol >;

    public:
      Linear( );
      Linear( const TRow& t );

      template< class _TNumber, class ... _TArgs >
      Linear( const _TNumber& v, _TArgs... args )
        : m_W( nullptr )
        {
          if( sizeof...( args ) > 0 )
          {
            Self n( args... );
            this->SetParameters( TRow::Zero( sizeof...( args ) + 1 ) );
            this->SetWeights( n.GetParameters( ) );
            this->SetBias( TScalar( v ) );
          }
          else
          {
            TRow p( 1 );
            p( 0, 0 ) = TScalar( v );
            this->SetParameters( p );
          } // end if
        }

      virtual ~Linear( );

      void AnalyticalFit( const TMatrix& X, const TCol& y );

      const unsigned long& GetDimensions( ) const;
      TRow GetWeights( ) const;
      TScalar GetBias( ) const;
      const TRow& GetParameters( ) const;

      virtual void Init( unsigned long n, const PUJ::EInitValues& e );

      void SetWeights( const TRow& w );
      void SetBias( const TScalar& b );
      void SetParameters( const TRow& t );

      virtual TScalar operator()( const TRow& x ) const;
      virtual TCol operator()( const TMatrix& x ) const;

    protected:
      void _StreamIn( std::istream& i );
      void _StreamOut( std::ostream& o ) const;

    protected:
      unsigned long m_N;
      TRow   m_Parameters;
      _TCol* m_W;

    public:

      /**
       */
      class PUJ_ML_EXPORT Cost
        : public _TBaseCost
      {
      public:
        Cost( );
        virtual ~Cost( ) = default;

        virtual TScalar operator()(
          unsigned int i, TRow* g = nullptr
          ) const override;
        virtual void operator-=( const TRow& g ) override;
        virtual void SetTrainData( const TMatrix& X, const TMatrix& Y ) override;

      protected:
        std::vector< TMatrix > m_XtX;
        std::vector< TMatrix > m_uX;
        std::vector< TMatrix > m_Xy;
        std::vector< TScalar > m_uy;
        std::vector< TScalar > m_yty;
      };

    public:
      friend std::istream& operator>>( std::istream& i, Self& m )
        {
          m._StreamIn( i );
          return( i );
        }
      friend std::ostream& operator<<( std::ostream& o, const Self& m )
        {
          m._StreamOut( o );
          return( o );
        }
    };
  } // end namespace
} // end namespace

#endif // __PUJ__Linear__h__

// eof - $RCSfile$
